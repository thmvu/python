# Bai scikit-learn: Hepatitis Logistic Regression

Thu muc nay thuc hanh preprocessing du lieu Hepatitis bang scikit-learn va
su dung Logistic Regression de phan loai cot `class`.

## Chay bai thuc hanh

```powershell
cd baiskikitlearn
python hepatitis_logistic_regression.py
```

Neu may chua co thu vien can thiet:

```powershell
pip install -r requirements.txt
```

## Noi dung preprocessing

- Doc du lieu tu URL GitHub raw.
- Chuyen cac gia tri rong thanh missing value.
- Tach `class` lam nhan can du doan.
- Chuyen cac cot so ve numeric va dien missing value bang median.
- Xu ly cac cot categorical/boolean bang most-frequent imputer va one-hot encoding.
- Scale cac cot so bang `StandardScaler`.
- Huan luyen model `LogisticRegression`.
- In accuracy, confusion matrix va classification report.
