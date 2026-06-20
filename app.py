from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier
from src.pipline.training_pipeline import TrainPipeline

app = FastAPI()

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request = request

        self.Gender = None
        self.Age = None
        self.Driving_License = None
        self.Region_Code = None
        self.Previously_Insured = None
        self.Annual_Premium = None
        self.Policy_Sales_Channel = None
        self.Vintage = None
        self.Vehicle_Age_lt_1_Year = None
        self.Vehicle_Age_gt_2_Years = None
        self.Vehicle_Damage_Yes = None

    async def get_vehicle_data(self):
        form = await self.request.form()

        self.Gender = form.get("Gender")
        self.Age = form.get("Age")
        self.Driving_License = form.get("Driving_License")
        self.Region_Code = form.get("Region_Code")
        self.Previously_Insured = form.get("Previously_Insured")
        self.Annual_Premium = form.get("Annual_Premium")
        self.Policy_Sales_Channel = form.get("Policy_Sales_Channel")
        self.Vintage = form.get("Vintage")
        self.Vehicle_Age_lt_1_Year = form.get("Vehicle_Age_lt_1_Year")
        self.Vehicle_Age_gt_2_Years = form.get("Vehicle_Age_gt_2_Years")
        self.Vehicle_Damage_Yes = form.get("Vehicle_Damage_Yes")


@app.get("/")
async def index(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="vehicledata.html",
        context={"context": "Rendering"},
    )


@app.get("/train")
async def train_route():

    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()

        return Response("Training Successful!")

    except Exception as e:
        return Response(f"Error Occurred: {str(e)}")


@app.post("/")
async def predict_route(request: Request):

    try:
        form = DataForm(request)
        await form.get_vehicle_data()

        vehicle_data = VehicleData(
            Gender=form.Gender,
            Age=form.Age,
            Driving_License=form.Driving_License,
            Region_Code=form.Region_Code,
            Previously_Insured=form.Previously_Insured,
            Annual_Premium=form.Annual_Premium,
            Policy_Sales_Channel=form.Policy_Sales_Channel,
            Vintage=form.Vintage,
            Vehicle_Age_lt_1_Year=form.Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_Years=form.Vehicle_Age_gt_2_Years,
            Vehicle_Damage_Yes=form.Vehicle_Damage_Yes,
        )

        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        predictor = VehicleDataClassifier()

        prediction = predictor.predict(vehicle_df)[0]

        status = "Response-Yes" if prediction == 1 else "Response-No"

        return templates.TemplateResponse(
            request=request,
            name="vehicledata.html",
            context={"context": status},
        )

    except Exception as e:
        return {
            "status": False,
            "error": str(e)
        }


if __name__ == "__main__":
    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT
    )