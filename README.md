# modeling_air_quality_fire_machine_learning


----------------------------
Goal: predict air quality based on wildfire, weather, location, and time


----------------------------
Key Questions for the Future:

How does fire proximity affect PM2.5 levels?

Which pollutants are most impacted by wildfires?

Do weather conditions moderate fire impacts on air quality?

Can we cluster regions by air quality response patterns?


----------------------------
Models Used:

Linear Regression – predict PM2.5 from fire distance and conditions, weather

KNN – predict air quality from similar conditions

K-Means Clustering – cluster monitoring stations by pollution patterns, interpret clusters (e.g., heavily impacted, moderately impacted)

PCA – reduce pollutants to 2-3 principal components

----------------------------
Numeric Variables (14+):

PM2.5, PM10, O3, NO2, CO, SO2

Weather – Humidity, temperature, wind speed, wind direction, precipitation

Distance to fire

Fire brightness, confidence, FRP

Fires within radii (50/100/200km)

Fire intensity score

Day of week, month

----------------------------
Categorical Variables (6+):

Fire proximity category (Very Close/Close/Moderate/Far)

PM2.5 category (Good/Moderate/Unhealthy...)

Has nearby fire (Yes/No)

Weekend (Yes/No)

Station name (if you keep it)

Month (can treat as categorical)








