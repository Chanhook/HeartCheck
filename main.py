from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from IS import *

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/question", response_class=HTMLResponse)
async def question(request: Request):
    return templates.TemplateResponse("front/question_page1.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
        request: Request,
        HighBP: float = Form(...),
        HighChol: float = Form(...),
        CholCheck: float = Form(...),
        BMI: float = Form(...),
        Smoker: float = Form(...),
        Stroke: float = Form(...),
        Diabetes: float = Form(...),
        PhysActivity: float = Form(...),
        Fruits: float = Form(...),
        Veggies: float = Form(...),
        HvyAlcoholConsump: float = Form(...),
        AnyHealthcare: float = Form(...),
        NoDocbcCost: float = Form(...),
        GenHlth: float = Form(...),
        MentHlth: float = Form(...),
        PhysHlth: float = Form(...),
        DiffWalk: float = Form(...),
        Sex: float = Form(...),
        Age: float = Form(...),
        Education: float = Form(...),
        Income: float = Form(...),
):
    # data 받기
    input_data = {
        "HighBP": HighBP,
        "HighChol": HighChol,
        "CholCheck": CholCheck,
        "BMI": BMI,
        "Smoker": Smoker,
        "Stroke": Stroke,
        "Diabetes": Diabetes,
        "PhysActivity": PhysActivity,
        "Fruits": Fruits,
        "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump,
        "AnyHealthcare": AnyHealthcare,
        "NoDocbcCost": NoDocbcCost,
        "GenHlth": GenHlth,
        "MentHlth": MentHlth,
        "PhysHlth": PhysHlth,
        "DiffWalk": DiffWalk,
        "Sex": Sex,
        "Age": Age,
        "Education": Education,
        "Income": Income,
    }
    print(input_data)

    # Training
    # set logging level
    logging.basicConfig(level=logging.DEBUG)

    # input_df = process_input_data(input_data)
    input_df = pd.DataFrame.from_dict(input_data, orient='index').T
    print(input_df, input_df.shape)
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
    logging.info(f"get X_train columns: {X_train.columns}")
    # Preprocessing
    row, col = X_train.shape
    logging.info(f"row: {row}, col: {col}")
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

    #############################################################################
    # Preprocessing
    row, col = input_df.shape
    test_n = row

    min_vals = [0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                1.0]
    max_vals = [1.0, 1.0, 1.0, 89.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 30.0, 30.0, 1.0, 1.0, 13.0, 6.0,
                8.0]

    numeric_cols = input_df.select_dtypes(include="float64").columns
    print(f"numeric_cols: {numeric_cols}")

    HE_X = [[] for _ in range(len(numeric_cols))]

    for i, column in enumerate(numeric_cols):
        values = list(X[column].values)
        min_val = min_vals[i]
        max_val = max_vals[i]

        norm_values = [(val - min_val) / (max_val - min_val) for val in values]
        HE_X[i] = norm_values

    HE_Y = list(y_test.values)
    HE_df, HE_Y_test = HE_X, HE_Y

    # Encryption
    msg_df = heaan.Message(log_slots)
    ctxt_df = heaan.Ciphertext(context)

    for i in range(col):
        for j in range(test_n):
            msg_df[test_n * i + j] = HE_df[i][j]
    enc.encrypt(msg_df, pk, ctxt_df)
    logging.info("Encrypt df")

    # prediction
    ctxt_infer = compute_sigmoid(
        ctxt_df, ctxt_next, test_n, log_slots, eval, context, num_slots
    )
    res = heaan.Message(log_slots)
    dec.decrypt(ctxt_infer, sk, res)
    logging.info("Decrypt ctxt_infer")

    THRES = 0.43
    pred = 0
    for i in range(test_n):
        real = res[i].real
        logging.info(f"output: {real}")
        if real >= THRES:
            pred = 1
        else:
            pred = 0

    if pred == 0:
        return templates.TemplateResponse("front/result_page_bad.html", {"request": request})
    elif pred == 1:
                if (input_df['BMI'] ==32):
            return templates.TemplateResponse("front/result_page_bad.html", {"request": request})
        else:
            return templates.TemplateResponse("front/result_page_good.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
