from pathlib import Path
import io
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import FancyArrowPatch
import streamlit as st

from catboost import Pool


def configure_plot_font():
    preferred_fonts = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "Noto Sans CJK KR",
        "NanumGothic",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]

    installed_fonts = {
        font.name
        for font in font_manager.fontManager.ttflist
    }

    for font_name in preferred_fonts:
        if font_name in installed_fonts:
            rcParams["font.family"] = font_name
            rcParams["axes.unicode_minus"] = False
            return font_name

    rcParams["axes.unicode_minus"] = False
    return None


PLOT_FONT = configure_plot_font()


# 1. PAGE SETUP

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
)


# 2. Languages and interface text

LANGUAGES = ['🇬🇧 English', "🇺🇿 O'zbek", '🇰🇷 한국어']

I18N = {'🇬🇧 English': {'title': 'Heart Disease Prediction 🫀',
                'subtitle': 'Frozen thesis model — research demo',
                'verified': '✓ Inference verified on 61,755 protected final-test rows',
                'warning': 'For research demonstration only. This app is not a clinical diagnostic '
                           'tool.',
                'language': 'Language',
                'model_info': 'Model information',
                'model': 'Model',
                'model_value': 'Frozen CatBoost pipeline',
                'calibration': 'Calibration',
                'calibration_value': 'Isotonic regression',
                'threshold': 'Frozen threshold',
                'predictors': 'Predictors',
                'no_training': 'No training, tuning, threshold selection, or calibration fitting '
                               'occurs inside this app.',
                'replay_intro': 'Replay intro',
                'loading': ['Analyzing...', 'Loading model...', 'Preparing...', 'Almost ready...'],
                'basic': '1. Basic profile',
                'health': '2. Health history',
                'lifestyle': '3. Lifestyle',
                'run': 'Run Prediction',
                'result': 'Prediction result',
                'model_decision': 'Model decision',
                'flagged': 'FLAGGED',
                'not_flagged': 'NOT FLAGGED',
                'case_above': "This case is above the frozen model's decision threshold.",
                'case_below': "This case is below the frozen model's decision threshold.",
                'model_probability': 'Model Estimated Probability',
                'probability_note': 'Calibrated model estimate for the positive class. This is not '
                                    'a diagnosis.',
                'input_summary': 'Submitted input summary',
                'feature': 'Feature',
                'entered_value': 'Entered value',
                'chart_title': 'Local SHAP Waterfall — Individual Prediction',
                'chart_xlabel': 'CatBoost raw margin / log-odds',
                'base_output': 'Base output',
                'final_output': 'Final output',
                'full_shap': 'View full local SHAP contribution table',
                'shap_influence': 'SHAP influence',
                'direction': 'Direction',
                'direction_positive': 'Toward FLAGGED',
                'direction_negative': 'Toward NOT FLAGGED',
                'direction_neutral': 'Neutral',
                'shap_base': 'SHAP base value',
                'shap_final_margin': 'Final CatBoost raw margin',
                'shap_error': 'SHAP additivity error',
                'shap_pass': 'Local SHAP additivity check: PASS',
                'shap_review': 'Local SHAP additivity check requires review.',
                'download_shap': 'Download Full Local SHAP PNG (300 dpi)',
                'download_pub_shap': 'Download Publication SHAP PNG (300 dpi)',
                'download_csv': 'Download prediction result (.csv)',
                'footer': 'Streamlit research demonstration | Frozen model, threshold, '
                          'calibration, and feature schema | Inference integrity verified across '
                          '61,755 protected final-test rows.'},
 "🇺🇿 O'zbek": {'title': 'Yurak Kasalligi Bashorati 🫀',
               'subtitle': 'Muzlatilgan thesis modeli — tadqiqot demo',
               'verified': '✓ Inference 61,755 ta final-test qatorda tekshirildi',
               'warning': "Faqat tadqiqot namoyishi uchun. Bu ilova klinik tashxis qo'yish "
                          'vositasi emas.',
               'language': 'Til',
               'model_info': 'Model haqida',
               'model': 'Model',
               'model_value': 'Muzlatilgan CatBoost pipeline',
               'calibration': 'Kalibratsiya',
               'calibration_value': 'Isotonic regression',
               'threshold': 'Muzlatilgan threshold',
               'predictors': 'Predictorlar',
               'no_training': "Bu ilova ichida model qayta o'qitilmaydi, tuning qilinmaydi, "
                              'threshold yoki calibration qayta tanlanmaydi.',
               'replay_intro': "Intro'ni qayta ko'rish",
               'loading': ['Tahlil qilinmoqda...',
                           'Model yuklanmoqda...',
                           'Tayyorlanmoqda...',
                           'Deyarli tayyor...'],
               'basic': "1. Asosiy ma'lumotlar",
               'health': "2. Sog'liq tarixi",
               'lifestyle': '3. Turmush tarzi',
               'run': 'Bashoratni ishga tushirish',
               'result': 'Bashorat natijasi',
               'model_decision': 'Model qarori',
               'flagged': 'FLAGGED',
               'not_flagged': 'NOT FLAGGED',
               'case_above': "Model score muzlatilgan threshold'dan yuqori.",
               'case_below': "Model score muzlatilgan threshold'dan past.",
               'model_probability': 'Model Estimated Probability',
               'probability_note': 'Positive class uchun kalibratsiya qilingan model bahosi. Bu '
                                   'tibbiy tashxis emas.',
               'input_summary': 'Kiritilgan qiymatlar',
               'feature': 'Ustun',
               'entered_value': 'Kiritilgan qiymat',
               'chart_title': 'Local SHAP Waterfall — Individual Bashorat',
               'chart_xlabel': 'CatBoost raw margin / log-odds',
               'base_output': "Boshlang'ich output",
               'final_output': 'Yakuniy output',
               'full_shap': "To'liq local SHAP jadvalini ko'rish",
               'shap_influence': "SHAP ta'siri",
               'direction': "Yo'nalish",
               'direction_positive': 'FLAGGED tomonga',
               'direction_negative': 'NOT FLAGGED tomonga',
               'direction_neutral': 'Neytral',
               'shap_base': 'SHAP base value',
               'shap_final_margin': 'Yakuniy CatBoost raw margin',
               'shap_error': 'SHAP additivity xatosi',
               'shap_pass': 'Local SHAP additivity check: PASS',
               'shap_review': 'Local SHAP additivity check tekshirishni talab qiladi.',
               'download_shap': "To'liq Local SHAP PNG yuklab olish (300 dpi)",
               'download_pub_shap': 'Publication SHAP PNG yuklab olish (300 dpi)',
               'download_csv': 'Bashorat natijasini yuklab olish (.csv)',
               'footer': 'Streamlit research demonstration | Frozen model, threshold, calibration, '
                         'and feature schema | Inference integrity verified across 61,755 '
                         'protected final-test rows.'},
 '🇰🇷 한국어': {'title': '심장 질환 예측 시스템 🫀',
            'subtitle': '고정된 논문 모델 — 연구 데모',
            'verified': '✓ 61,755개 보호된 최종 테스트 행에서 추론 검증 완료',
            'warning': '연구용 데모입니다. 이 앱은 임상 진단 도구가 아닙니다.',
            'language': '언어',
            'model_info': '모델 정보',
            'model': '모델',
            'model_value': '고정된 CatBoost 파이프라인',
            'calibration': '보정',
            'calibration_value': 'Isotonic regression',
            'threshold': '고정 임계값',
            'predictors': '예측 변수',
            'no_training': '이 앱에서는 재학습, 튜닝, 임계값 선택 또는 보정 학습을 수행하지 않습니다.',
            'replay_intro': '인트로 다시 보기',
            'loading': ['분석 중...', '모델 로딩...', '준비 중...', '거의 완료...'],
            'basic': '1. 기본 정보',
            'health': '2. 건강 이력',
            'lifestyle': '3. 생활 습관',
            'run': '예측 실행',
            'result': '예측 결과',
            'model_decision': '모델 결정',
            'flagged': 'FLAGGED',
            'not_flagged': 'NOT FLAGGED',
            'case_above': '이 사례는 고정된 결정 임계값보다 높습니다.',
            'case_below': '이 사례는 고정된 결정 임계값보다 낮습니다.',
            'model_probability': 'Model Estimated Probability',
            'probability_note': '양성 클래스에 대한 보정된 모델 추정치입니다. 의학적 진단이 아닙니다.',
            'input_summary': '입력값 요약',
            'feature': '변수',
            'entered_value': '입력값',
            'chart_title': 'Local SHAP Waterfall — 개별 예측',
            'chart_xlabel': 'CatBoost raw margin / log-odds',
            'base_output': '기준 출력',
            'final_output': '최종 출력',
            'full_shap': '전체 Local SHAP 표 보기',
            'shap_influence': 'SHAP 영향',
            'direction': '방향',
            'direction_positive': 'FLAGGED 방향',
            'direction_negative': 'NOT FLAGGED 방향',
            'direction_neutral': '중립',
            'shap_base': 'SHAP 기준값',
            'shap_final_margin': '최종 CatBoost raw margin',
            'shap_error': 'SHAP additivity 오차',
            'shap_pass': 'Local SHAP additivity check: PASS',
            'shap_review': 'Local SHAP additivity check 검토 필요',
            'download_shap': '전체 Local SHAP PNG 다운로드 (300 dpi)',
            'download_pub_shap': 'Publication SHAP PNG 다운로드 (300 dpi)',
            'download_csv': '예측 결과 다운로드 (.csv)',
            'footer': 'Streamlit research demonstration | Frozen model, threshold, calibration, '
                      'and feature schema | Inference integrity verified across 61,755 protected '
                      'final-test rows.'}}


FEATURE_LABELS = {'🇬🇧 English': {'Height_(cm)': 'Height (cm)',
                'Weight_(kg)': 'Weight (kg)',
                'BMI': 'BMI',
                'Alcohol_Consumption': 'Alcohol consumption',
                'Fruit_Consumption': 'Fruit consumption',
                'Green_Vegetables_Consumption': 'Green vegetables consumption',
                'Fried_Potato_Consumption': 'Fried potato consumption',
                'General_Health': 'General health',
                'Checkup': 'Checkup',
                'Age_Category': 'Age category',
                'Exercise': 'Exercise',
                'Skin_Cancer': 'Skin cancer',
                'Other_Cancer': 'Other cancer',
                'Depression': 'Depression',
                'Diabetes': 'Diabetes',
                'Arthritis': 'Arthritis',
                'Sex': 'Sex',
                'Smoking_History': 'Smoking history'},
 "🇺🇿 O'zbek": {'Height_(cm)': "Bo'y (sm)",
               'Weight_(kg)': 'Vazn (kg)',
               'BMI': 'BMI',
               'Alcohol_Consumption': "Spirtli ichimlik iste'moli",
               'Fruit_Consumption': "Meva iste'moli",
               'Green_Vegetables_Consumption': "Yashil sabzavot iste'moli",
               'Fried_Potato_Consumption': "Qovurilgan kartoshka iste'moli",
               'General_Health': "Umumiy sog'liq holati",
               'Checkup': "Tibbiy ko'rik",
               'Age_Category': 'Yosh toifasi',
               'Exercise': 'Jismoniy mashq',
               'Skin_Cancer': 'Teri saratoni',
               'Other_Cancer': 'Boshqa saraton',
               'Depression': 'Depressiya',
               'Diabetes': 'Diabet',
               'Arthritis': 'Artrit',
               'Sex': 'Jinsi',
               'Smoking_History': 'Chekish tarixi'},
 '🇰🇷 한국어': {'Height_(cm)': '키 (cm)',
            'Weight_(kg)': '체중 (kg)',
            'BMI': 'BMI',
            'Alcohol_Consumption': '음주량',
            'Fruit_Consumption': '과일 섭취',
            'Green_Vegetables_Consumption': '녹색 채소 섭취',
            'Fried_Potato_Consumption': '튀긴 감자 섭취',
            'General_Health': '전반적 건강 상태',
            'Checkup': '건강검진',
            'Age_Category': '연령대',
            'Exercise': '운동',
            'Skin_Cancer': '피부암',
            'Other_Cancer': '기타 암',
            'Depression': '우울증',
            'Diabetes': '당뇨병',
            'Arthritis': '관절염',
            'Sex': '성별',
            'Smoking_History': '흡연 이력'}}


CATEGORY_TRANSLATIONS = {'🇬🇧 English': {},
 "🇺🇿 O'zbek": {'Yes': 'Ha',
               'No': "Yo'q",
               'Male': 'Erkak',
               'Female': 'Ayol',
               'Poor': 'Yomon',
               'Fair': 'Qoniqarli',
               'Good': 'Yaxshi',
               'Very Good': 'Juda yaxshi',
               'Excellent': "A'lo",
               'Never': 'Hech qachon',
               '5 or more years ago': "5 yoki undan ko'p yil oldin",
               'Within the past 5 years': "So'nggi 5 yil ichida",
               'Within the past 2 years': "So'nggi 2 yil ichida",
               'Within the past year': "So'nggi 1 yil ichida",
               'No, pre-diabetes or borderline diabetes': "Yo'q, prediabet yoki chegaraviy diabet",
               'Yes, but female told only during pregnancy': 'Ha, lekin faqat homiladorlik davrida '
                                                             'aytilgan'},
 '🇰🇷 한국어': {'Yes': '예',
            'No': '아니오',
            'Male': '남성',
            'Female': '여성',
            'Poor': '나쁨',
            'Fair': '보통 이하',
            'Good': '좋음',
            'Very Good': '매우 좋음',
            'Excellent': '우수',
            'Never': '없음',
            '5 or more years ago': '5년 이상 전',
            'Within the past 5 years': '최근 5년 이내',
            'Within the past 2 years': '최근 2년 이내',
            'Within the past year': '최근 1년 이내',
            'No, pre-diabetes or borderline diabetes': '아니오, 당뇨 전단계 또는 경계성 당뇨',
            'Yes, but female told only during pregnancy': '예, 여성의 경우 임신 중에만 진단'}}


# 3. SESSION STATE + SPLASH

if "lang" not in st.session_state:
    st.session_state.lang = "🇬🇧 English"

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if not st.session_state.loaded:
    splash = st.empty()
    icons = ["🫀", "🩺", "💓", "🩸"]
    splash_texts = I18N[st.session_state.lang]["loading"]

    for icon, loading_text in zip(
        icons,
        splash_texts,
    ):
        with splash.container():
            st.markdown(
                f"""
                <div style="
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    height:55vh;
                ">
                    <div style="font-size:76px; margin-bottom:8px;">
                        {icon}
                    </div>
                    <h1 style="text-align:center; margin:8px 0 4px;">
                        Heart Disease Prediction
                    </h1>
                    <p style="
                        color:#64748b;
                        font-size:15px;
                        text-align:center;
                    ">
                        {loading_text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        time.sleep(0.25)

    splash.empty()
    st.session_state.loaded = True
    st.rerun()


# 4. PATHS

APP_DIR = Path(__file__).resolve().parent
METADATA_PATH = APP_DIR / "stage12_metadata.json"
MODEL_PATH = APP_DIR / "final_catboost_pipeline.joblib"
CALIBRATOR_PATH = APP_DIR / "final_isotonic_calibrator.joblib"
SCHEMA_PATH = APP_DIR / "input_schema.json"


# 5. FROZEN FEATURE SET

FEATURES = [
    "Height_(cm)",
    "Weight_(kg)",
    "BMI",
    "Alcohol_Consumption",
    "Fruit_Consumption",
    "Green_Vegetables_Consumption",
    "Fried_Potato_Consumption",
    "General_Health",
    "Checkup",
    "Age_Category",
    "Exercise",
    "Skin_Cancer",
    "Other_Cancer",
    "Depression",
    "Diabetes",
    "Arthritis",
    "Sex",
    "Smoking_History",
]

AGE_ORDER = [
    "18-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80+",
]

GENERAL_HEALTH_ORDER = [
    "Poor",
    "Fair",
    "Good",
    "Very Good",
    "Excellent",
]

CHECKUP_ORDER = [
    "Never",
    "5 or more years ago",
    "Within the past 5 years",
    "Within the past 2 years",
    "Within the past year",
]

NOMINAL_FEATURES = [
    "Exercise",
    "Skin_Cancer",
    "Other_Cancer",
    "Depression",
    "Diabetes",
    "Arthritis",
    "Sex",
    "Smoking_History",
]


# 6. LOAD FROZEN ARTIFACTS

@st.cache_resource
def load_artifacts():

    with open(
        METADATA_PATH,
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    threshold = float(
        metadata["threshold"]
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(
            f"Calibrator file not found: {CALIBRATOR_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    return model, calibrator, threshold


@st.cache_data
def load_input_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as file:
        return json.load(file)


model, calibrator, threshold = load_artifacts()
input_schema = load_input_schema()


# 7. Small translation helpers

def t(key):
    return I18N[
        st.session_state.lang
    ][key]


def feature_label(feature):
    return FEATURE_LABELS[
        st.session_state.lang
    ][feature]


def category_label(value):
    return CATEGORY_TRANSLATIONS[
        st.session_state.lang
    ].get(
        str(value),
        str(value),
    )


def format_input_value(
    feature,
    value,
):

    if pd.isna(value):
        return "Missing"

    if feature in (
        "General_Health",
        "Checkup",
        "Exercise",
        "Skin_Cancer",
        "Other_Cancer",
        "Depression",
        "Diabetes",
        "Arthritis",
        "Sex",
        "Smoking_History",
    ):
        return category_label(
            value
        )

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        value = float(
            value
        )

        if value.is_integer():
            return str(
                int(value)
            )

        return f"{value:.2f}"

    return str(value)


# 8. Input helpers

def numeric_input(feature):
    info = input_schema[feature]
    return st.number_input(
        feature_label(feature),
        min_value=float(info["min"]),
        max_value=float(info["max"]),
        value=float(info["default"]),
        step=0.1,
        key=f"num_{feature}",
    )


def category_options(feature, preferred_order=None):
    observed = input_schema[feature]["options"]
    if preferred_order is None:
        return observed
    return [value for value in preferred_order if value in observed]


def category_default_index(feature, options):
    default_value = input_schema[feature]["default"]
    return options.index(default_value) if default_value in options else 0


def category_input(
    feature,
    preferred_order=None,
):

    options = category_options(
        feature,
        preferred_order,
    )

    return st.selectbox(
        feature_label(
            feature
        ),
        options,
        index=category_default_index(
            feature,
            options,
        ),
        format_func=category_label,
        key=f"cat_{feature}",
    )


def calibrate(
    raw_score,
):

    score = np.array(
        [raw_score]
    )

    if hasattr(
        calibrator,
        "predict",
    ):
        return float(
            calibrator.predict(
                score
            )[0]
        )

    return float(
        calibrator.transform(
            score
        )[0]
    )


# 9. Local SHAP helpers

def original_feature_for_transformed(
    transformed_name,
):

    if transformed_name in FEATURES:
        return transformed_name

    for feature in sorted(
        NOMINAL_FEATURES,
        key=len,
        reverse=True,
    ):
        if transformed_name.startswith(
            feature + "_"
        ):
            return feature

    raise ValueError(
        "Could not map transformed feature "
        f"to original predictor: {transformed_name}"
    )


def local_grouped_shap(
    input_data,
):

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    catboost_model = model.named_steps[
        "model"
    ]

    transformed = preprocessor.transform(
        input_data
    )

    transformed_names = list(
        preprocessor
        .get_feature_names_out()
    )

    shap_full = (
        catboost_model
        .get_feature_importance(
            data=Pool(
                transformed,
                feature_names=transformed_names,
            ),
            type="ShapValues",
        )
    )

    transformed_shap = shap_full[
        0,
        :-1,
    ]

    expected_value = float(
        shap_full[
            0,
            -1,
        ]
    )

    grouped = {
        feature: 0.0
        for feature in FEATURES
    }

    for name, value in zip(
        transformed_names,
        transformed_shap,
    ):
        original = (
            original_feature_for_transformed(
                name
            )
        )

        grouped[
            original
        ] += float(
            value
        )

    model_margin = float(
        np.asarray(
            catboost_model.predict(
                transformed,
                prediction_type="RawFormulaVal",
            )
        ).reshape(-1)[0]
    )

    shap_margin = (
        expected_value
        + sum(
            grouped.values()
        )
    )

    additivity_error = abs(
        model_margin
        - shap_margin
    )

    rows = []

    for feature in FEATURES:

        contribution = grouped[
            feature
        ]

        if contribution > 0:
            direction_key = (
                "direction_positive"
            )
        elif contribution < 0:
            direction_key = (
                "direction_negative"
            )
        else:
            direction_key = (
                "direction_neutral"
            )

        rows.append(
            {
                "Feature key": feature,
                "Input value": input_data.iloc[
                    0
                ][feature],
                "SHAP contribution": contribution,
                "Direction key": direction_key,
                "Absolute contribution": abs(
                    contribution
                ),
            }
        )

    explanation = pd.DataFrame(
        rows
    )

    explanation = (
        explanation
        .sort_values(
            "Absolute contribution",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        explanation,
        expected_value,
        model_margin,
        additivity_error,
    )


def make_shap_waterfall(
    explanation,
    expected_value,
    model_margin,
    probability_threshold,
    top_n=10,
):

    ranked = (
        explanation
        .sort_values(
            "Absolute contribution",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    top = (
        ranked
        .head(
            top_n
        )
        .copy()
    )

    remainder = (
        ranked
        .iloc[
            top_n:
        ]
        .copy()
    )

    if not remainder.empty:

        other_contribution = float(
            remainder[
                "SHAP contribution"
            ].sum()
        )

        other_row = pd.DataFrame(
            [{
                "Feature key": "__OTHER__",
                "Input value": (
                    f"{len(remainder)} predictors"
                ),
                "SHAP contribution": other_contribution,
                "Direction key": (
                    "direction_positive"
                    if other_contribution > 0
                    else (
                        "direction_negative"
                        if other_contribution < 0
                        else "direction_neutral"
                    )
                ),
                "Absolute contribution": abs(
                    other_contribution
                ),
            }]
        )

        plot_data = pd.concat(
            [
                top,
                other_row,
            ],
            ignore_index=True,
        )

    else:
        plot_data = top.copy()

    threshold_margin = float(
        np.log(
            probability_threshold
            / (
                1.0
                - probability_threshold
            )
        )
    )

    # Keep the waterfall additive.
    current = float(
        expected_value
    )

    cumulative_points = [
        current
    ]

    for contribution in plot_data[
        "SHAP contribution"
    ]:
        current += float(
            contribution
        )

        cumulative_points.append(
            current
        )

    all_x = (
        cumulative_points
        + [
            float(
                model_margin
            ),
            threshold_margin,
        ]
    )

    x_min = min(
        all_x
    )

    x_max = max(
        all_x
    )

    x_span = max(
        x_max - x_min,
        1e-6,
    )

    padding = (
        x_span
        * 0.08
    )

    fig, ax = plt.subplots(
        figsize=(
            9.2,
            6.2,
        )
    )

    ax.set_xlim(
        x_min - padding,
        x_max + padding,
    )

    positive_color = "#ff0051"
    negative_color = "#008bfb"

    labels = []
    current = float(
        expected_value
    )

    for row_number, row in plot_data.iterrows():

        feature = row[
            "Feature key"
        ]

        contribution = float(
            row[
                "SHAP contribution"
            ]
        )

        next_value = (
            current
            + contribution
        )

        if feature == "__OTHER__":

            label = (
                f"Other {len(remainder)} predictors"
            )

        else:

            entered_value = (
                format_input_value(
                    feature,
                    row[
                        "Input value"
                    ],
                )
            )

            # Publication figure stays in English for thesis consistency.
            english_feature = FEATURE_LABELS[
                "🇬🇧 English"
            ][feature]

            label = (
                f"{entered_value} = "
                f"{english_feature}"
            )

        labels.append(
            label
        )

        color = (
            positive_color
            if contribution >= 0
            else negative_color
        )

        arrow = FancyArrowPatch(
            (
                current,
                row_number,
            ),
            (
                next_value,
                row_number,
            ),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=9,
            color=color,
            shrinkA=0,
            shrinkB=0,
            capstyle="butt",
        )

        ax.add_patch(
            arrow
        )

        if abs(
            contribution
        ) >= (
            x_span
            * 0.085
        ):

            ax.text(
                (
                    current
                    + next_value
                )
                / 2,
                row_number,
                f"{contribution:+.3f}",
                va="center",
                ha="center",
                color="white",
                fontsize=8.5,
                fontweight="bold",
            )

        else:

            offset = (
                x_span
                * 0.012
            )

            ax.text(
                next_value
                + (
                    offset
                    if contribution >= 0
                    else -offset
                ),
                row_number,
                f"{contribution:+.3f}",
                va="center",
                ha=(
                    "left"
                    if contribution >= 0
                    else "right"
                ),
                color=color,
                fontsize=8.5,
                fontweight="bold",
            )

        if row_number < (
            len(
                plot_data
            )
            - 1
        ):

            ax.plot(
                [
                    next_value,
                    next_value,
                ],
                [
                    row_number + 0.20,
                    row_number + 0.80,
                ],
                linestyle="--",
                linewidth=0.9,
                alpha=0.35,
            )

        current = (
            next_value
        )

    y_positions = np.arange(
        len(
            plot_data
        )
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        labels,
        fontsize=9.3,
    )

    ax.invert_yaxis()

    # Three reference lines. The exact values are shown outside the figure
    # in Streamlit cards, so the graph itself stays uncluttered.
    ax.axvline(
        expected_value,
        linestyle="--",
        linewidth=1.2,
        alpha=0.65,
    )

    ax.axvline(
        threshold_margin,
        linestyle="--",
        linewidth=1.5,
        color="#f59e0b",
        alpha=0.9,
    )

    ax.axvline(
        model_margin,
        linestyle="-",
        linewidth=1.4,
        color="#111827",
        alpha=0.95,
    )

    ax.set_title(
        "Local SHAP Waterfall — Individual Prediction",
        fontsize=15,
        fontweight="bold",
        pad=34,
    )

    ax.set_xlabel(
        "CatBoost raw model output (log-odds)",
        labelpad=16,
        loc="center",
    )

    ax.grid(
        axis="x",
        alpha=0.12,
        linewidth=0.8,
    )

    ax.tick_params(
        axis="y",
        length=0,
        pad=2,
    )

    for spine in [
        "top",
        "right",
        "left",
    ]:
        ax.spines[
            spine
        ].set_visible(
            False
        )

    ax.spines[
        "bottom"
    ].set_alpha(
        0.35
    )

    # Compact reference-line legend, avoiding overlapping annotations.
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="#9ca3af", linestyle="--", linewidth=1.5,
               label=f"Baseline: {expected_value:.3f}"),
        Line2D([0], [0], color="#f59e0b", linestyle="--", linewidth=1.5,
               label=f"Threshold: {threshold_margin:.3f}"),
        Line2D([0], [0], color="#111827", linestyle="-", linewidth=1.5,
               label=f"Final output: {model_margin:.3f}"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=8.5,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.6,
    )

    fig.subplots_adjust(
        left=0.37,
        right=0.97,
        top=0.76,
        bottom=0.15,
    )

    png_buffer = io.BytesIO()

    fig.savefig(
        png_buffer,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    png_buffer.seek(
        0
    )

    return (
        fig,
        png_buffer.getvalue(),
        threshold_margin,
        len(
            remainder
        ),
    )


# 10. STYLING

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 760;
        margin-top: 0.5rem;
        margin-bottom: 6px;
        line-height: 1.2;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        margin-bottom: 10px;
    }

    .verified-wrap {
        text-align: center;
        margin-bottom: 24px;
    }

    .verified-badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        border: 1px solid #d1fae5;
        background: #ecfdf5;
        color: #166534;
        font-size: 13px;
        font-weight: 700;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .status-card {
        padding: 20px 22px;
        border-radius: 13px;
        margin-top: 10px;
        margin-bottom: 12px;
        font-weight: 760;
        font-size: 19px;
    }

    .status-negative {
        background: #ecfdf3;
        border: 1px solid #bbf7d0;
        color: #166534;
    }

    .status-positive {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        color: #b91c1c;
    }

    .probability-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 22px;
        background: #ffffff;
        text-align: center;
        min-height: 122px;
    }

    .probability-label {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .probability-value {
        font-size: 42px;
        font-weight: 820;
        line-height: 1.1;
    }

    .plain-note {
        color: #475569;
        font-size: 14px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    div[data-testid="stFormSubmitButton"] button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        width: 100%;
        min-height: 46px;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #e03b3b;
        color: white;
        border: none;
    }

    div[data-testid="stDownloadButton"] button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 11. SIDEBAR

with st.sidebar:

    selected_language = st.selectbox(
        "Language / Til / 언어",
        LANGUAGES,
        index=LANGUAGES.index(
            st.session_state.lang
        ),
    )

    if (
        selected_language
        != st.session_state.lang
    ):
        st.session_state.lang = (
            selected_language
        )

        st.rerun()

    st.divider()

    st.header(
        t(
            "model_info"
        )
    )

    st.write(
        f"**{t('model')}:** "
        f"{t('model_value')}"
    )

    st.write(
        f"**{t('calibration')}:** "
        f"{t('calibration_value')}"
    )

    st.write(
        f"**{t('threshold')}:** "
        f"{threshold:.6f}"
    )

    st.write(
        f"**{t('predictors')}:** 18"
    )

    st.divider()

    st.caption(
        t(
            "no_training"
        )
    )

    if st.button(
        t(
            "replay_intro"
        ),
        use_container_width=True,
    ):
        st.session_state.loaded = False
        st.rerun()


# 12. TITLE

st.markdown(
    (
        "<div class='main-title'>"
        f"{t('title')}"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.markdown(
    (
        "<div class='subtitle'>"
        f"{t('subtitle')}"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.markdown(
    (
        "<div class='verified-wrap'>"
        "<span class='verified-badge'>"
        f"{t('verified')}"
        "</span>"
        "</div>"
    ),
    unsafe_allow_html=True,
)

st.warning(
    t(
        "warning"
    )
)


# 13. INPUT FORM

with st.form(
    "heart_disease_form"
):

    st.markdown(
        (
            "<div class='section-title'>"
            f"{t('basic')}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:
        age = category_input(
            "Age_Category",
            AGE_ORDER,
        )

    with c2:
        sex = category_input(
            "Sex",
        )

    with c3:
        general_health = category_input(
            "General_Health",
            GENERAL_HEALTH_ORDER,
        )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:
        height = numeric_input(
            "Height_(cm)"
        )

    with c2:
        weight = numeric_input(
            "Weight_(kg)"
        )

    with c3:
        bmi = numeric_input(
            "BMI"
        )

    st.markdown(
        (
            "<div class='section-title'>"
            f"{t('health')}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:
        checkup = category_input(
            "Checkup",
            CHECKUP_ORDER,
        )

    with c2:
        diabetes = category_input(
            "Diabetes",
        )

    with c3:
        arthritis = category_input(
            "Arthritis",
        )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:
        depression = category_input(
            "Depression",
        )

    with c2:
        skin_cancer = category_input(
            "Skin_Cancer",
        )

    with c3:
        other_cancer = category_input(
            "Other_Cancer",
        )

    st.markdown(
        (
            "<div class='section-title'>"
            f"{t('lifestyle')}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:
        exercise = category_input(
            "Exercise",
        )

    with c2:
        smoking = category_input(
            "Smoking_History",
        )

    with c3:
        alcohol = numeric_input(
            "Alcohol_Consumption"
        )

    c1, c2, c3 = st.columns(
        3
    )

    with c1:
        fruit = numeric_input(
            "Fruit_Consumption"
        )

    with c2:
        vegetables = numeric_input(
            "Green_Vegetables_Consumption"
        )

    with c3:
        fried_potato = numeric_input(
            "Fried_Potato_Consumption"
        )

    st.divider()

    submitted = st.form_submit_button(
        t(
            "run"
        ),
        use_container_width=True,
    )


# 14. PREDICTION + EXPLANATION

if submitted:

    input_data = pd.DataFrame(
        [{
            "Height_(cm)": height,
            "Weight_(kg)": weight,
            "BMI": bmi,
            "Alcohol_Consumption": alcohol,
            "Fruit_Consumption": fruit,
            "Green_Vegetables_Consumption": vegetables,
            "Fried_Potato_Consumption": fried_potato,
            "General_Health": general_health,
            "Checkup": checkup,
            "Age_Category": age,
            "Exercise": exercise,
            "Skin_Cancer": skin_cancer,
            "Other_Cancer": other_cancer,
            "Depression": depression,
            "Diabetes": diabetes,
            "Arthritis": arthritis,
            "Sex": sex,
            "Smoking_History": smoking,
        }]
    )

    input_data = input_data[
        FEATURES
    ]

    transformed_input = model.named_steps["preprocessor"].transform(input_data)
    raw_score = float(model.named_steps["model"].predict_proba(transformed_input)[0, 1])

    prediction = int(
        raw_score
        >= threshold
    )

    calibrated_probability = (
        calibrate(
            raw_score
        )
    )

    threshold_distance = (
        raw_score
        - threshold
    )

    st.divider()

    st.subheader(
        t(
            "result"
        )
    )

    status_class = (
        "status-positive"
        if prediction == 1
        else "status-negative"
    )

    decision_text = (
        t(
            "flagged"
        )
        if prediction == 1
        else t(
            "not_flagged"
        )
    )

    r1, r2 = st.columns(
        [
            1,
            1,
        ]
    )

    with r1:

        st.markdown(
            f"""
            <div class="status-card {status_class}">
                {t('model_decision')}: {decision_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            (
                "<div class='plain-note'>"
                f"{t('case_above') if prediction == 1 else t('case_below')}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    with r2:

        st.markdown(
            f"""
            <div class="probability-card">
                <div class="probability-label">
                    {t('model_probability')}
                </div>
                <div class="probability-value">
                    {calibrated_probability:.2%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            (
                "<div class='plain-note' style='text-align:center;'>"
                f"{t('probability_note')}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )



    with st.expander(
        t(
            "input_summary"
        )
    ):

        input_summary = pd.DataFrame(
            {
                t("feature"): [
                    feature_label(
                        feature
                    )
                    for feature in FEATURES
                ],
                t("entered_value"): [
                    format_input_value(
                        feature,
                        input_data.iloc[
                            0
                        ][feature],
                    )
                    for feature in FEATURES
                ],
            }
        )

        st.dataframe(
            input_summary,
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # Local SHAP
    # -----------------------------------------------------


    (
        local_shap_table,
        shap_base_value,
        shap_model_margin,
        shap_additivity_error,
    ) = local_grouped_shap(
        input_data
    )

    (
        publication_fig,
        publication_png,
        shap_threshold_margin,
        other_count,
    ) = make_shap_waterfall(
        local_shap_table,
        shap_base_value,
        shap_model_margin,
        threshold,
        top_n=10,
    )

    st.pyplot(
        publication_fig,
        use_container_width=True,
    )

    plt.close(
        publication_fig
    )


    with st.expander(
        t(
            "full_shap"
        )
    ):

        shap_display = pd.DataFrame(
            {
                t("feature"): [
                    feature_label(
                        feature
                    )
                    for feature in local_shap_table[
                        "Feature key"
                    ]
                ],
                t("entered_value"): [
                    format_input_value(
                        feature,
                        value,
                    )
                    for feature, value in zip(
                        local_shap_table[
                            "Feature key"
                        ],
                        local_shap_table[
                            "Input value"
                        ],
                    )
                ],
                t("shap_influence"): (
                    local_shap_table[
                        "SHAP contribution"
                    ]
                    .round(
                        6
                    )
                    .tolist()
                ),
                t("direction"): [
                    t(
                        direction_key
                    )
                    for direction_key in local_shap_table[
                        "Direction key"
                    ]
                ],
            }
        )

        st.dataframe(
            shap_display,
            use_container_width=True,
            hide_index=True,
        )

        st.write(
            f"**{t('shap_base')}:** "
            f"{shap_base_value:.6f}"
        )

        st.write(
            f"**{t('shap_final_margin')}:** "
            f"{shap_model_margin:.6f}"
        )

        st.write(
            f"**{t('shap_error')}:** "
            f"{shap_additivity_error:.3e}"
        )

        if (
            shap_additivity_error
            < 1e-8
        ):
            st.success(
                t(
                    "shap_pass"
                )
            )
        else:
            st.warning(
                t(
                    "shap_review"
                )
            )

    # -----------------------------------------------------
    # Downloads
    # -----------------------------------------------------

    result_export = (
        input_data.copy()
    )

    result_export[
        "raw_model_score"
    ] = raw_score

    result_export[
        "frozen_threshold"
    ] = threshold

    result_export[
        "calibrated_probability"
    ] = calibrated_probability

    result_export[
        "prediction"
    ] = prediction

    result_export[
        "model_decision"
    ] = (
        "FLAGGED"
        if prediction == 1
        else "NOT FLAGGED"
    )

    d1, d2 = st.columns(
        2
    )

    with d1:
        st.download_button(
            "Download publication SHAP PNG (300 dpi)",
            data=publication_png,
            file_name=(
                "local_shap_publication_top10_plus_other.png"
            ),
            mime="image/png",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            t(
                "download_csv"
            ),
            data=result_export.to_csv(
                index=False
            ),
            file_name=(
                "heart_disease_prediction_result.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


# 15. FOOTER

st.divider()

st.caption(
    t(
        "footer"
    )
)

