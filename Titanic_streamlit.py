import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump
from fpdf import FPDF
import base64
from tempfile import NamedTemporaryFile
import os

st.set_page_config(page_title="Titanic ML Dashboard", layout="wide")
st.title("🚢 Titanic Survival Prediction - ML Dashboard")
st.markdown("""
    ### 👇 ارفع ملف Titanic CSV وابدأ التحليل
    سيتم تدريب نموذج تعلم آلي لتوقع من سيبقى على قيد الحياة.
""")

# 1. تحميل البيانات
uploaded_file = st.file_uploader("📁 ارفع ملف CSV الخاص ببيانات Titanic", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 نظرة على البيانات")
    st.dataframe(df.head())

    # 📊 صفحة التحليل الاستكشافي
    st.markdown("---")
    st.subheader("📈 التحليل الاستكشافي للبيانات (EDA)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### توزيع الأعمار")
        fig_age, ax = plt.subplots()
        sns.histplot(df["Age"].dropna(), kde=True, ax=ax)
        st.pyplot(fig_age)

    with col2:
        st.markdown("#### توزيع الأسعار (Fare)")
        fig_fare, ax = plt.subplots()
        sns.histplot(df["Fare"], kde=True, ax=ax)
        st.pyplot(fig_fare)

    fig_class, ax = plt.subplots()
    sns.countplot(data=df, x="Pclass", hue="Survived", ax=ax)
    ax.set_title("عدد الناجين حسب الدرجة")
    st.pyplot(fig_class)

    # 2. المعالجة المسبقة
    df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]].dropna()
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])

    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. اختيار النموذج
    model_option = st.radio("🔍 اختر النموذج الذي ترغب في استخدامه:", ("Decision Tree", "Logistic Regression"))

    if model_option == "Decision Tree":
        st.markdown("#### ⚙️ تحسين النموذج باستخدام GridSearchCV")
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        st.success(f"✅ أفضل إعدادات: {grid.best_params_}")
    else:
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)

    # 4. التنبؤ والتقييم
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("🎯 دقة النموذج", f"{acc:.2f}")

    with st.expander("📊 تقرير التصنيف"):
        st.text(classification_report(y_test, y_pred))

    with st.expander("🔵 مصفوفة الالتباس"):
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig_cm)

    with st.expander("🟠 منحنى ROC"):
        if model_option == "Logistic Regression":
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig_roc)

    # 5. حفظ النموذج
    model_filename = "model.joblib"
    with st.expander("💾 حفظ النموذج"):
        if st.button("💾 احفظ النموذج الحالي"):
            dump(model, model_filename)
            st.success(f"تم حفظ النموذج باسم {model_filename}")

    # 📌 صفحة مقارنة النماذج
    st.markdown("---")
    st.subheader("📊 مقارنة بين Decision Tree و Logistic Regression")
    dt_model = DecisionTreeClassifier().fit(X_train, y_train)
    lr_model = LogisticRegression(solver='liblinear').fit(X_train, y_train)

    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test))

    comp_df = pd.DataFrame({
        'النموذج': ['Decision Tree', 'Logistic Regression'],
        'الدقة': [dt_acc, lr_acc]
    })
    st.table(comp_df)

    st.markdown("""
    - Decision Tree مفيد في البيانات التي تحتوي على علاقات غير خطية واضحة.
    - Logistic Regression بسيط وفعال في التنبؤات الثنائية إذا كانت البيانات خطية.
    """)

    # ✅ صفحة ختامية
    st.markdown("---")
    st.subheader("📥 ملخص المشروع وتحميل النموذج")
    st.markdown("""
    هذا المشروع يعرض استخدام تقنيات تعلم الآلة لتحليل بيانات Titanic.
    قمنا بـ:
    - استكشاف البيانات وتحليلها بصريًا
    - بناء نموذجين (شجرة القرار والانحدار اللوجستي)
    - مقارنة الأداء وتفسير النتائج

    🔽 يمكنك تحميل النموذج المدرب لاستخدامه لاحقًا:
    """)
    if os.path.exists(model_filename):
        with open(model_filename, "rb") as f:
            btn = st.download_button(
                label="📥 تحميل النموذج",
                data=f,
                file_name=model_filename,
                mime="application/octet-stream"
            )
