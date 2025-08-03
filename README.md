# 🚢 Titanic Survival Prediction - Machine Learning Dashboard

تطبيق تفاعلي باستخدام Python و Streamlit لتحليل بيانات Titanic وتوقع من سيبقى على قيد الحياة باستخدام تقنيات تعلم الآلة.

---

## 📌 أهداف المشروع
- استكشاف البيانات بصريًا (EDA)
- تدريب نموذجين: Decision Tree و Logistic Regression
- مقارنة الأداء بين النماذج
- عرض الرسوم البيانية التفاعلية
- تصدير النتائج كنموذج جاهز `.joblib`
- واجهة مستخدم تفاعلية باستخدام Streamlit

---

## 🛠️ الأدوات والتقنيات
- Python (pandas, scikit-learn, matplotlib, seaborn)
- Streamlit
- joblib لحفظ النموذج
- FPDF + base64 لتوليد تقرير PDF (اختياري)

---

## 🚀 تشغيل المشروع

```bash
pip install -r requirements.txt
streamlit run Titanic_streamlit.py

