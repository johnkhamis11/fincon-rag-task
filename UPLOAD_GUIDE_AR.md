تنظيم المستودع والنشر — Yassa Hany
=================================

المستودع: https://github.com/Yassahany2004/fincon-rag-task

ارفع المجلدات نفسها مع الملفات الموجودة بجوارها. لا تجمع الملفات الموجودة داخل المجلدات في الجذر؛ التطبيق يعتمد على أسماء هذه المجلدات.

يجب أن تظهر finrag وdata وreports وtests و.streamlit و.github كمجلدات في الصفحة الرئيسية للمستودع، ويظهر streamlit_app.py بجوارها مباشرة.

الملفات الموجودة حاليًا في جذر المستودع تحتاج نقلها للمجلدات الصحيحة. لترتيب النسخة الحالية بدون فقد الملفات: افتح المستودع ثم اضغط النقطة في لوحة المفاتيح لفتح github.dev. أنشئ المجلدات وانقل الملفات حسب الخريطة أدناه. استبدل streamlit_app.py وREADME.md بالنسختين المصححتين من هذا الملف المضغوط. ثم سجّل التغييرات من Source Control بحسابك. تقدر أيضًا رفع المجلدات الجديدة بالسحب والإفلات من داخل مجلد المشروع، ثم حذف النسخ القديمة الموجودة في الجذر حتى لا تتكرر الملفات.


خريطة نقل الملفات الموجودة حاليًا في الجذر:

- __init__.py → finrag/__init__.py
- documents.py → finrag/documents.py
- engine.py → finrag/engine.py
- generation.py → finrag/generation.py
- evaluation.py → finrag/evaluation.py
- baai_financial_corpus.json → data/baai_financial_corpus.json
- financial_regulatory_standards_2026.json → data/financial_regulatory_standards_2026.json
- benchmark.json → data/benchmark.json
- browser-report.json → reports/browser-report.json
- evaluation-report.json → reports/evaluation-report.json
- groq-live-report.json → reports/groq-live-report.json
- live-deployment-report.json → reports/live-deployment-report.json
- preview-app.png → reports/preview-app.png
- test_pipeline.py → tests/test_pipeline.py
- test_streamlit.py → tests/test_streamlit.py
- config.toml → .streamlit/config.toml
- secrets.toml.example → .streamlit/secrets.toml.example
- ci.yml → .github/workflows/ci.yml

خطوات النشر:
1. افتح https://share.streamlit.io/deploy?repository=Yassahany2004/fincon-rag-task&branch=main&mainModule=streamlit_app.py
2. Repository: Yassahany2004/fincon-rag-task
3. Branch: main
4. Main file path: streamlit_app.py
5. Advanced settings → Python 3.12.
6. أضف مفتاحك في Secrets باسم GROQ_API_KEY، واستخدم GROQ_MODEL = "openai/gpt-oss-20b".
7. Deploy، ثم افتح التطبيق وتأكد أن Answer mode يظهر Groq.
8. جرّب سؤال Form 8-K، وافتح Evidence trail. اختبر HyDE وMulti-query من Sidebar.

المفتاح الحقيقي مكانه Streamlit Secrets، وليس ملفات GitHub.
الروابط وتقارير الاختبار السابقة داخل reports تخص النشر السابق؛ جرّب الرابط الجديد بعد اكتمال نشره.
