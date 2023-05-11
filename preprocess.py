import pandas as pd

from input_data import InputData


def make_dataframe(input_list):
    # input_list를 데이터프레임으로 만들기
    columns = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'Diabetes', 'PhysActivity', 'Fruits',
               'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',
               'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    df = pd.DataFrame([input_list], columns=columns)
    return df


def process_input_data(input_data: InputData) -> pd.DataFrame:
    input_dict = input_data.dict()
    df = pd.DataFrame([input_dict])
    return df


if __name__ == "__main__":
    input_list = [1.0, 1.0, 1.0, 40.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 5.0, 18.0, 15.0, 1.0, 0.0, 9.0, 4.0,
                  3.0]
    dataframe = make_dataframe(input_list)
    print(dataframe)

    input_data = InputData(
        HighBP=1.0,
        HighChol=1.0,
        CholCheck=1.0,
        BMI=40.0,
        Smoker=1.0,
        Stroke=0.0,
        Diabetes=0.0,
        PhysActivity=0.0,
        Fruits=0.0,
        Veggies=1.0,
        HvyAlcoholConsump=0.0,
        AnyHealthcare=1.0,
        NoDocbcCost=0.0,
        GenHlth=5.0,
        MentHlth=18.0,
        PhysHlth=15.0,
        DiffWalk=1.0,
        Sex=0.0,
        Age=9.0,
        Education=4.0,
        Income=3.0
    )
    df = process_input_data(input_data)
    print(df)
    [1.0,
     1.0,
     1.0,
     1.0,
     30.0,
     1.0,
     0.0,
     2.0,
     0.0,
     1.0,
     1.0,
     0.0,
     1.0,
     0.0,
     5.0,
     30.0,
     30.0,
     1.0,
     0.0,
     9.0,
     5.0,
     1.0]
