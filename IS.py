import logging
import os
import random

import numpy as np
import pandas as pd
import piheaan as heaan
from piheaan.math import approx
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def set_random_seed(seed):
    """Set the random seed for numpy and random.

    Args:
        seed (int): The random seed to use.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)


def balanced_sampling(df, target_col, ratio):
    """
    Perform balanced sampling on a pandas dataframe based on a target column and a desired ratio.

    Args:
        df (pandas.DataFrame): The dataframe to sample from.
        target_col (str): The target column to balance the sampling on.
        ratio (float): The desired ratio of samples with value 0 to samples with value 1.

    Returns:
        pandas.DataFrame: The sampled dataframe.
    """
    df["HeartDiseaseorAttack"].value_counts()

    # 1. Distinguish between indices with a value of 0 in the target column and indices with a value of 1.
    zero_idx = df[df[target_col] == 0].index
    one_idx = df[df[target_col] == 1].index[:400]

    # 2. Select the fewer number of indices with a value of 0 and indices with a value of 1.
    # 3. The ratio you select is set to match the ratio of 0s to 1s.
    zero_sample_size = int(len(one_idx) / ratio)  # 0 sample size
    zero_sample_idx = np.random.choice(
        zero_idx, size=zero_sample_size, replace=False
    )  # 0 sample index

    # 3. Create a new dataframe with the dataset containing the selected indexes.
    sampled_idx = np.concatenate([one_idx, zero_sample_idx])  # chosen index
    sampled_df = df.loc[sampled_idx]  # chosen dataframe

    # 4. Convert the target column to integers.
    sampled_df[target_col] = sampled_df[target_col].astype(int)

    return sampled_df


def generate_key(key_file_path="./keys", params=heaan.ParameterPreset.FGb):
    """
    Generates HEAAN public and secret keys and returns them along with other necessary objects.

    Args:
        - key_file_path (str): path to the directory where the keys will be saved
        - params (heaan.ParameterPreset): the parameter set to use for generating the keys

    Returns:
        A tuple containing:
        - context (heaan.Context): the HEAAN context object
        - sk (heaan.SecretKey): the HEAAN secret key object
        - pk (heaan.KeyPack): the HEAAN public key object
        - eval (heaan.HomEvaluator): the HEAAN homomorphic evaluator object
        - dec (heaan.Decryptor): the HEAAN decryptor object
        - enc (heaan.Encryptor): the HEAAN encryptor object
        - log_slots (int): the logarithm of the number of slots in the ciphertext
        - num_slots (int): the number of slots in the ciphertext
    """
    context = heaan.make_context(params)
    heaan.make_bootstrappable(context)

    if not os.path.exists(key_file_path):
        sk = heaan.SecretKey(context)
        os.makedirs(key_file_path, mode=0o775, exist_ok=True)
        sk.save(key_file_path + "/secretkey.bin")

        key_generator = heaan.KeyGenerator(context, sk)
        key_generator.gen_common_keys()
        key_generator.save(key_file_path + "/")

    sk = heaan.SecretKey(context, key_file_path + "/secretkey.bin")
    pk = heaan.KeyPack(context, key_file_path + "/")
    pk.load_enc_key()
    pk.load_mult_key()

    eval = heaan.HomEvaluator(context, pk)
    dec = heaan.Decryptor(context)
    enc = heaan.Encryptor(context)

    log_slots = 15
    num_slots = 2 ** log_slots

    return context, sk, pk, eval, dec, enc, log_slots, num_slots


def preprocess_data(X, y):
    """Normalize the numeric columns of X and return the HE-ready data and labels.

    Args:
        X (pd.DataFrame): The features DataFrame.
        y (pd.Series): The target variable.

    Returns:
        tuple: A tuple containing:
            - HE_X (List[List[float]]): HE-ready features.
            - HE_Y (List[int]): The target variable.
    """
    numeric_cols = X.select_dtypes(include="float64").columns
    HE_X = [[] for _ in range(len(numeric_cols))]

    min_vals = []
    max_vals = []

    for i, column in enumerate(numeric_cols):
        values = list(X[column].values)
        min_val = min(values)
        max_val = max(values)

        min_vals.append(min_val)
        max_vals.append(max_val)

        norm_values = [(val - min_val) / (max_val - min_val) for val in values]
        HE_X[i] = norm_values

    HE_Y = list(y.values)

    logging.info(f"min vals: {min_vals}")
    logging.info(f"max vals: {max_vals}")

    return HE_X, HE_Y


def normalize_data(arr):
    """
    Normalize a 1-dimensional numpy array to have unit sum.

    Parameters:
        arr (numpy array): A 1-dimensional numpy array to be normalized.

    Returns:
        numpy array: The normalized version of the input array.
    """
    S = 0
    for i in range(len(arr)):
        S += arr[i]
    return [arr[i] / S for i in range(len(arr))]


def step(learning_rate, ctxt_X, ctxt_Y, ctxt_beta, n, log_slots, context, eval, num_slots):
    """
    ctxt_X, ctxt_Y : data for training
    ctxt_beta : initial value beta
    n : the number of row in train_data
    """
    ctxt_rot = heaan.Ciphertext(context)
    ctxt_tmp = heaan.Ciphertext(context)

    ## step1
    # beta0
    ctxt_beta0 = heaan.Ciphertext(context)
    eval.left_rotate(ctxt_beta, 8 * n, ctxt_beta0)

    # compute  ctxt_tmp = beta1*x1 + beta2*x2 + ... + beta8*x8 + beta0
    ctxt_tmp = heaan.Ciphertext(context)
    eval.mult(ctxt_beta, ctxt_X, ctxt_tmp)

    for i in range(3):
        eval.left_rotate(ctxt_tmp, n * 2 ** (2 - i), ctxt_rot)
        eval.add(ctxt_tmp, ctxt_rot, ctxt_tmp)
    eval.add(ctxt_tmp, ctxt_beta0, ctxt_tmp)

    msg_mask = heaan.Message(log_slots)
    for i in range(n):
        msg_mask[i] = 1
    eval.mult(ctxt_tmp, msg_mask, ctxt_tmp)

    ## step2
    # compute sigmoid
    approx.sigmoid(eval, ctxt_tmp, ctxt_tmp, 8.0)
    eval.bootstrap(ctxt_tmp, ctxt_tmp)
    msg_mask = heaan.Message(log_slots)
    # if sigmoid(0) -> return 0.5
    for i in range(n, num_slots):
        msg_mask[i] = 0.5
    eval.sub(ctxt_tmp, msg_mask, ctxt_tmp)

    ## step3
    # compute  (learning_rate/n) * (y_(j) - p_(j))
    ctxt_d = heaan.Ciphertext(context)
    eval.sub(ctxt_Y, ctxt_tmp, ctxt_d)
    eval.mult(ctxt_d, learning_rate / n, ctxt_d)

    eval.right_rotate(ctxt_d, 8 * n, ctxt_tmp)  # for beta0
    for i in range(3):
        eval.right_rotate(ctxt_d, n * 2 ** i, ctxt_rot)
        eval.add(ctxt_d, ctxt_rot, ctxt_d)
    eval.add(ctxt_d, ctxt_tmp, ctxt_d)

    ## step4
    # compute  (learning_rate/n) * (y_(j) - p_(j)) * x_(j)
    ctxt_X_j = heaan.Ciphertext(context)
    msg_X0 = heaan.Message(log_slots)
    for i in range(8 * n, 9 * n):
        msg_X0[i] = 1
    eval.add(ctxt_X, msg_X0, ctxt_X_j)
    eval.mult(ctxt_X_j, ctxt_d, ctxt_d)

    ## step5
    # compute  Sum_(all j) (learning_rate/n) * (y_(j) - p_(j)) * x_(j)
    for i in range(9):
        eval.left_rotate(ctxt_d, 2 ** (8 - i), ctxt_rot)
        eval.add(ctxt_d, ctxt_rot, ctxt_d)
    msg_mask = heaan.Message(log_slots)
    for i in range(9):
        msg_mask[i * n] = 1
    eval.mult(ctxt_d, msg_mask, ctxt_d)

    for i in range(9):
        eval.right_rotate(ctxt_d, 2 ** i, ctxt_rot)
        eval.add(ctxt_d, ctxt_rot, ctxt_d)

    ## step6
    # update beta
    eval.add(ctxt_beta, ctxt_d, ctxt_d)
    return ctxt_d


def compute_sigmoid(ctxt_X, ctxt_beta, n, log_slots, eval, context, num_slots):
    """
    ctxt_X : data for evaluation
    ctxt_beta : estimated beta from function 'step'
    n : the number of row in test_data
    """
    ctxt_rot = heaan.Ciphertext(context)
    ctxt_tmp = heaan.Ciphertext(context)

    # beta0
    ctxt_beta0 = heaan.Ciphertext(context)
    eval.left_rotate(ctxt_beta, 8 * n, ctxt_beta0)

    # compute x * beta + beta0
    ctxt_tmp = heaan.Ciphertext(context)
    eval.mult(ctxt_beta, ctxt_X, ctxt_tmp)

    for i in range(3):
        eval.left_rotate(ctxt_tmp, n * 2 ** (2 - i), ctxt_rot)
        eval.add(ctxt_tmp, ctxt_rot, ctxt_tmp)
    eval.add(ctxt_tmp, ctxt_beta0, ctxt_tmp)

    msg_mask = heaan.Message(log_slots)
    for i in range(n):
        msg_mask[i] = 1
    eval.mult(ctxt_tmp, msg_mask, ctxt_tmp)

    # compute sigmoid
    approx.sigmoid(eval, ctxt_tmp, ctxt_tmp, 8.0)
    eval.bootstrap(ctxt_tmp, ctxt_tmp)
    msg_mask = heaan.Message(log_slots)
    for i in range(n, num_slots):
        msg_mask[i] = 0.5
    eval.sub(ctxt_tmp, msg_mask, ctxt_tmp)

    return ctxt_tmp


if __name__ == "__main__":
    # set logging level
    logging.basicConfig(level=logging.DEBUG)

    # set random seed

    seed = 34
    set_random_seed(seed)
    logging.info(f"set seed number {seed}")

    # read csv
    df = pd.read_csv("heart_disease_health_indicators.csv")
    sampled_df = balanced_sampling(df, "HeartDiseaseorAttack", 0.5)
    logging.info(
        "Data has been sampled with balanced sampling method. The target class is now balanced with a ratio of 0.5."
    )

    # generate HE key
    context, sk, pk, eval, dec, enc, log_slots, num_slots = generate_key()
    logging.info("key has been generated.")

    # Train Test split
    X = sampled_df.drop(["HeartDiseaseorAttack"], axis=1)
    y = sampled_df["HeartDiseaseorAttack"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=34
    )
    logging.info("create train dataset, test dataset")

    # Preprocessing
    row, col = X_train.shape
    train_n = row
    HE_X, HE_Y = preprocess_data(X_train, y_train)
    logging.info("Train dataset has been preprocessed")

    # Encryption
    msg_X = heaan.Message(log_slots)
    ctxt_X = heaan.Ciphertext(context)
    for i in range(col):
        for j in range(train_n):
            msg_X[train_n * i + j] = HE_X[i][j]
    enc.encrypt(msg_X, pk, ctxt_X)
    logging.info("Encrypt X_train")

    msg_Y = heaan.Message(log_slots)
    ctxt_Y = heaan.Ciphertext(context)
    for j in range(train_n):
        msg_Y[j] = HE_Y[j]
    enc.encrypt(msg_Y, pk, ctxt_Y)
    logging.info("Encrypt y_train")

    # initial value beta
    beta = 2 * np.random.rand(col + 1) - 1
    logging.info(f"Initial beta : {beta}")

    msg_beta = heaan.Message(log_slots)
    ctxt_beta = heaan.Ciphertext(context)

    for i in range(col):
        for j in range(train_n):
            msg_beta[train_n * i + j] = beta[i + 1]
    for j in range(train_n):
        msg_beta[col * train_n + j] = beta[0]

    enc.encrypt(msg_beta, pk, ctxt_beta)
    logging.info(f"Msg beta : {msg_beta}")

    # randomly assign learning_rate
    learning_rate = 0.2
    num_steps = 200
    ctxt_next = heaan.Ciphertext(context)
    eval.add(ctxt_beta, 0, ctxt_next)
    for i in tqdm(range(num_steps), total=num_steps):
        # estimate beta_hat using function 'step' for 100 iteration
        ctxt_next = step(
            0.2, ctxt_X, ctxt_Y, ctxt_next, train_n, log_slots, context, eval, num_slots
        )
    logging.info("Training is done")

    # Preprocessing
    row, col = X_test.shape
    test_n = row
    HE_X_test, HE_Y_test = preprocess_data(X_test, y_test)
    logging.info("Test dataset has been preprocessed")

    # Encryption
    msg_X_test = heaan.Message(log_slots)
    ctxt_X_test = heaan.Ciphertext(context)
    for i in range(col):
        for j in range(test_n):
            msg_X_test[test_n * i + j] = HE_X_test[i][j]
    enc.encrypt(msg_X_test, pk, ctxt_X_test)
    logging.info("Encrypt X_test")

    # evalutaion
    ctxt_infer = compute_sigmoid(
        ctxt_X_test, ctxt_next, test_n, log_slots, eval, context, num_slots
    )
    res = heaan.Message(log_slots)
    dec.decrypt(ctxt_infer, sk, res)
    logging.info("Decrypt ctxt_infer")

    # find best f1 score
    best_f1_score = 0
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_THRES = 0

    for THRES in np.arange(0, 1.01, 0.01):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(test_n):
            real = res[i].real
            logging.info(f"output: {real}")
            if real >= THRES:
                if HE_Y_test[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if HE_Y_test[i] == 0:
                    TN += 1
                else:
                    FN += 1

        accuracy = (TP + TN) / test_n
        # Exception Handling
        try:
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = TP / (TP + FN)
        except ZeroDivisionError:
            recall = 0
        try:
            f1_score = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_THRES = THRES

            print("New best f1 score found: {:.4f}".format(f1_score))
            print(
                "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, THRES: {:.2f}".format(
                    accuracy, precision, recall, THRES
                )
            )

    print("\nBest THRES: {:.2f}".format(best_THRES))
    print(
        "Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(
            best_accuracy, best_precision, best_recall, best_f1_score
        )
    )
