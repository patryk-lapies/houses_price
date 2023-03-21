import pandas as pd
import numpy as np
import string


class Preprocessing:

    def __init__(self, data_path: string):
        self.data_path = data_path

    def preprocess(self, only_numeric=False) -> pd.DataFrame:
        """Function preprocess given data for further usage, Only numeric parameter decides if function return
        only numeric columns before using get_dummies and mapping some categorical columns"""

        data = pd.read_csv(self.data_path, index_col='Id')

        data.drop('Alley', axis=1, inplace=True)
        data['MSSubClass'] = data['MSSubClass'].astype(str)

        # dropping columns with the lowest correlation with price
        columns_drop = ['BedroomAbvGr', 'ScreenPorch', 'PoolArea', 'MoSold', '3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath',
                        'MiscVal', 'LowQualFinSF', 'YrSold', 'OverallCond', 'EnclosedPorch', 'KitchenAbvGr']
        data = data.drop(columns_drop, axis=1)

        # filling missing values
        Lot_Frontage_mean = np.mean(data['LotFrontage'])
        data['LotFrontage'] = data['LotFrontage'].fillna(Lot_Frontage_mean)

        data = data[data['MasVnrType'].notna()]

        data['BsmtQual'] = data['BsmtQual'].fillna('None')

        data['BsmtCond'] = data['BsmtCond'].fillna('None')
        data['BsmtExposure'] = data['BsmtExposure'].fillna('None')
        data['BsmtFinType1'] = data['BsmtFinType1'].fillna('None')
        data['BsmtFinType2'] = data['BsmtFinType2'].fillna('None')

        data = data[data['Electrical'].notna()]

        data = data.drop(['GarageYrBlt', 'GarageFinish'], axis=1)

        # If only numeric is True return data with only numeric variables
        if only_numeric:
            return data.select_dtypes(include=[np.number])

        data['FireplaceQu'] = data['FireplaceQu'].fillna('None')
        data['GarageType'] = data['GarageType'].fillna('None')
        data['GarageQual'] = data['GarageQual'].fillna('None')
        data['GarageCond'] = data['GarageCond'].fillna('None')

        data['PoolQC'] = data['PoolQC'].fillna('None')

        data['Fence'] = data['Fence'].fillna('None')

        data['MiscFeature'] = data['MiscFeature'].fillna('None')

        # Mapping categorical columns to integers according to data description

        data['ExterQual'] = data['ExterQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        data['ExterCond'] = data['ExterCond'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        data['BsmtQual'] = data['BsmtQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        data['BsmtCond'] = data['BsmtCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        data['BsmtExposure'] = data['BsmtExposure'].map({'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})

        data['BsmtFinType1'] = data['BsmtFinType1'].map({'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5,
                                                         'GLQ': 6})

        data['BsmtFinType2'] = data['BsmtFinType2'].map({'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5,
                                                         'GLQ': 6})

        data['HeatingQC'] = data['HeatingQC'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        data['KitchenQual'] = data['KitchenQual'].map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        data['Functional'] = data['Functional'].map({'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6,
                                                     'Min1': 7, 'Typ': 8})

        data['FireplaceQu'] = data['FireplaceQu'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        data['GarageQual'] = data['GarageQual'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        data['GarageCond'] = data['GarageCond'].map({'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

        data['PoolQC'] = data['PoolQC'].map({'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

        # transforming categorical data using pandas get_dummies
        data = pd.get_dummies(data)

        return data

