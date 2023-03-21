from data_preprocessing import Preprocessing
from model import Model
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data_preprocess = Preprocessing('../data/train.csv')
    data = data_preprocess.preprocess()

    x_train, x_test, y_train, y_test = train_test_split(data.drop('SalePrice', axis=1), data['SalePrice'],
                                                        random_state=42)

    model = Model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(model.evaluate(y_test, predictions))
