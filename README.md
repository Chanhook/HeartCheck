# Before start
you should install the required packages.

`pip install -r requirements.txt`

# Run the fastAPI
you can run the following command.

`python main.py`

# console output
```shell
INFO:     Started server process [22276]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```
When you go to http://localhost:8000 to conduct a survey and submit, the model predicts the results and displays the results page.


# Run HE model
you can run the following command.

`python IS.py`

or you can just click the "RUN Button" in your IDE.

# Result
result will be shown in the terminal like this
```shell
INFO:root:create train dataset, test dataset
INFO:root:Train dataset has been preprocessed
INFO:root:Encrypt X_train
INFO:root:Encrypt y_train
INFO:root:Initial beta : [ 0.35841535 -0.92097484 -0.12989304 -0.71441379 -0.00685288 -0.8420549
  0.61054971  0.89157566  0.75798019  0.57838619  0.96014217  0.97158372
 -0.73098259 -0.41266215  0.52729364  0.98442154  0.62464871 -0.29581138
 -0.60965801 -0.91178373  0.49186056  0.48730698]
INFO:root:Msg beta : [ (-0.920975+0.000000j), (-0.920975+0.000000j), (-0.920975+0.000000j), (-0.920975+0.000000j), (-0.920975+0.000000j), ..., (0.000000+0.000000j), (0.000000+0.000000j), (0.000000+0.000000j), (0.000000+0.000000j), (0.000000+0.000000j) ]
100%|██████████| 200/200 [00:06<00:00, 31.37it/s]
INFO:root:Training is done
INFO:root:Test dataset has been preprocessed
INFO:root:Encrypt X_test
New best f1 score found: 0.4665
Accuracy: 0.3042, Precision: 0.3042, Recall: 1.0000, THRES: 0.00
New best f1 score found: 0.4679
Accuracy: 0.3083, Precision: 0.3054, Recall: 1.0000, THRES: 0.06
New best f1 score found: 0.4725
Accuracy: 0.3208, Precision: 0.3093, Recall: 1.0000, THRES: 0.07
New best f1 score found: 0.4771
Accuracy: 0.3333, Precision: 0.3133, Recall: 1.0000, THRES: 0.08
New best f1 score found: 0.4915
Accuracy: 0.3792, Precision: 0.3273, Recall: 0.9863, THRES: 0.09
New best f1 score found: 0.5091
Accuracy: 0.4375, Precision: 0.3465, Recall: 0.9589, THRES: 0.10
New best f1 score found: 0.5176
Accuracy: 0.4875, Precision: 0.3626, Recall: 0.9041, THRES: 0.11
New best f1 score found: 0.5200
Accuracy: 0.5000, Precision: 0.3672, Recall: 0.8904, THRES: 0.12
New best f1 score found: 0.5249
Accuracy: 0.5625, Precision: 0.3919, Recall: 0.7945, THRES: 0.16
New best f1 score found: 0.5346
Accuracy: 0.5792, Precision: 0.4028, Recall: 0.7945, THRES: 0.17
New best f1 score found: 0.5490
Accuracy: 0.6167, Precision: 0.4275, Recall: 0.7671, THRES: 0.18
New best f1 score found: 0.5503
Accuracy: 0.6458, Precision: 0.4483, Recall: 0.7123, THRES: 0.19

Best THRES: 0.19
Accuracy: 0.6458, Precision: 0.4483, Recall: 0.7123, F1 Score: 0.5503
INFO:root:Decrypt ctxt_infer
```

# Demonstration
<img width="1624" alt="image" src="https://github.com/Chanhook/HeartCheck/assets/76461625/07313c00-6495-4ef0-bd57-2261e047e248">
survey page

<img width="1458" alt="image" src="https://github.com/Chanhook/HeartCheck/assets/76461625/992acd93-7ab8-495b-966b-a17ca49c19d4">
good page

<img width="1500" alt="image" src="https://github.com/Chanhook/HeartCheck/assets/76461625/aba19e00-09fa-49a5-9638-d8331c106a2c">
bad page
