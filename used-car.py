from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

# Canonical feature order from the training config.
DEFAULT_FEATURE_ORDER = [
	"Age",
	"KM",
	"FuelType",
	"HP",
	"MetColor",
	"Automatic",
	"CC",
	"Doors",
	"Weight",
]


def _candidate_artifact_paths() -> list[tuple[Path, Path]]:
	"""Return local model/features candidates in priority order (latest first)."""
	base = Path(__file__).resolve().parent
	return [
		(
			base / "models" / "used_car_price_prediction_model_v2.joblib",
			base / "models" / "used_car_price_prediction_features_v2.joblib",
		),
		(
			base / "models" / "used_car_price_prediction_model_v1.joblib",
			base / "models" / "used_car_price_prediction_features_v1.joblib",
		),
	]


@st.cache_resource(show_spinner=False)
def load_artifacts():
	"""Load model and ordered feature list from the first valid artifact pair."""
	for model_path, features_path in _candidate_artifact_paths():
		if model_path.exists() and features_path.exists():
			model = joblib.load(model_path)
			features = joblib.load(features_path)
			return model, list(features), model_path

	searched = "\n".join([f"- {m}\n  {f}" for m, f in _candidate_artifact_paths()])
	raise FileNotFoundError(
		"No used-car model artifacts found. Checked:\n" + searched
	)


def build_input_form() -> dict:
	st.sidebar.header("Car Details")

	age = st.sidebar.number_input("Age (years)", min_value=0, max_value=50, value=5, step=1)
	km = st.sidebar.number_input("KM (mileage)", min_value=0, max_value=1_000_000, value=60_000, step=1_000)
	fuel_type = st.sidebar.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"], index=0)
	hp = st.sidebar.number_input("Horse Power (HP)", min_value=30, max_value=500, value=110, step=1)
	met_color = st.sidebar.selectbox("Metallic Color", options=[0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
	automatic = st.sidebar.selectbox("Automatic", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
	cc = st.sidebar.number_input("Engine CC", min_value=500, max_value=5000, value=1600, step=50)
	doors = st.sidebar.selectbox("Number of Doors", options=[2, 3, 4, 5], index=2)
	weight = st.sidebar.number_input("Weight (kg)", min_value=600, max_value=3000, value=1100, step=10)

	return {
		"Age": age,
		"KM": km,
		"FuelType": fuel_type,
		"HP": hp,
		"MetColor": met_color,
		"Automatic": automatic,
		"CC": cc,
		"Doors": doors,
		"Weight": weight,
	}


def main():
	st.title("Used Car Price Predictor")
	st.caption("Enter car details and get an estimated market price.")

	try:
		model, features, loaded_model_path = load_artifacts()
	except Exception as exc:
		st.error(f"Failed to load model artifacts: {exc}")
		st.info(
			"Place used-car model/feature files under models/ in this project "
			"before deploying to Streamlit Community Cloud."
		)
		return

	st.success(f"Loaded model: {loaded_model_path.name}")

	user_input = build_input_form()

	st.subheader("Input Summary")
	st.write(pd.DataFrame([user_input]))

	if st.button("Predict Price", type="primary"):
		try:
			pred_df = pd.DataFrame([user_input])

			# Use persisted feature order when available.
			selected_features = features if features else DEFAULT_FEATURE_ORDER
			pred_df = pred_df[selected_features]

			prediction = model.predict(pred_df)[0]
			st.subheader("Estimated Used Car Price")
			st.metric(label="Predicted Price", value=f"EUR {prediction:,.2f}")
		except KeyError as exc:
			st.error(
				"Feature mismatch between app input and model feature list. "
				f"Missing key: {exc}"
			)
		except Exception as exc:
			st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
	main()
