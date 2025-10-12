import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import httpx
from pydantic import Field, BaseModel

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class WeatherClimateFunctionConfig(FunctionBaseConfig, name="weather_climate"):
    """
    Weather and Climate data retrieval function using Open-Meteo API.
    Provides weather forecasts, historical weather patterns, and climate alerts.
    """
    forecast_url: str = Field(default="https://api.open-meteo.com/v1/forecast", description="Forecast API URL")
    archive_url: str = Field(default="https://archive-api.open-meteo.com/v1/archive", description="Historical Archive API URL")
    geocoding_url: str = Field(default="https://geocoding-api.open-meteo.com/v1", description="Geocoding API URL")
    timeout: int = Field(default=30, description="API request timeout in seconds")


# Response models
class CurrentWeather(BaseModel):
    temperature: float
    humidity: float
    rainfall: float
    wind_speed: float
    conditions: str


class ForecastDay(BaseModel):
    date: str
    min_temp: float
    max_temp: float
    rainfall_mm: float
    humidity: float
    conditions: str


class WeatherForecastResponse(BaseModel):
    location: str
    current: CurrentWeather
    forecast: List[ForecastDay]


class HistoricalDataPoint(BaseModel):
    date: str
    temperature: Optional[float] = None
    rainfall: Optional[float] = None
    humidity: Optional[float] = None


class HistoricalSummary(BaseModel):
    avg_temperature: Optional[float] = None
    total_rainfall: Optional[float] = None
    avg_humidity: Optional[float] = None


class HistoricalWeatherResponse(BaseModel):
    location: str
    period: str
    data: List[HistoricalDataPoint]
    summary: HistoricalSummary


class ClimateAlert(BaseModel):
    type: str
    severity: str
    title: str
    description: str
    start_time: str
    end_time: str
    recommendations: List[str]


class ClimateAlertsResponse(BaseModel):
    location: str
    active_alerts: List[ClimateAlert]


@register_function(config_type=WeatherClimateFunctionConfig)
async def weather_climate_function(
    config: WeatherClimateFunctionConfig, builder: Builder
):
    """
    Weather and Climate tool using Open-Meteo API 
    """
    
    async def geocode_location(location: str) -> tuple[float, float, str]:
        """Convert location name to coordinates using Open-Meteo geocoding."""
        # Check if already coordinates
        if ',' in location:
            try:
                parts = location.split(',')
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                return lat, lon, location
            except:
                pass
        
        # Geocode location name
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.get(
                f"{config.geocoding_url}/search",
                params={"name": location, "count": 1, "language": "en", "format": "json"}
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("results"):
                raise ValueError(f"Location '{location}' not found")
            
            result = data["results"][0]
            return result["latitude"], result["longitude"], result["name"]
    
    def get_weather_condition(weather_code: int) -> str:
        """Map WMO weather codes to descriptions."""
        conditions = {
            0: "clear sky",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "depositing rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            71: "slight snow",
            73: "moderate snow",
            75: "heavy snow",
            77: "snow grains",
            80: "slight rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "slight snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with slight hail",
            99: "thunderstorm with heavy hail"
        }
        return conditions.get(weather_code, "unknown")
    
    async def get_weather_forecast(
        location: str,
        days: int = 7,
        units: str = "celsius"
    ) -> WeatherForecastResponse:
        """
        Get current and upcoming weather data for a specific location.
        
        Args:
            location: Location name or coordinates (lat,lon)
            days: Number of days to forecast (default: 7, max: 16)
            units: Temperature units - "celsius" or "fahrenheit"
        """
        try:
            days = min(days, 16)  # Open-Meteo supports up to 16 days
            lat, lon, location_name = await geocode_location(location)
            
            temp_unit = "celsius" if units == "celsius" else "fahrenheit"
            
            logger.info(f"Fetching weather forecast for: {location_name} ({lat}, {lon})")
            
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                # Get forecast data
                response = await client.get(
                    config.forecast_url,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "current": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m", "weather_code"],
                        "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum", "weather_code"],
                        "timezone": "auto",
                        "temperature_unit": temp_unit,
                        "forecast_days": days
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Process current weather
                current_data = data["current"]
                current = CurrentWeather(
                    temperature=current_data["temperature_2m"],
                    humidity=current_data["relative_humidity_2m"],
                    rainfall=current_data.get("rain", 0.0),
                    wind_speed=current_data["wind_speed_10m"],
                    conditions=get_weather_condition(current_data["weather_code"])
                )
                
                # Process daily forecast
                daily_data = data["daily"]
                forecast = []
                for i in range(len(daily_data["time"])):
                    # Calculate average humidity from current (Open-Meteo doesn't provide daily avg)
                    avg_humidity = current_data["relative_humidity_2m"]
                    
                    forecast.append(ForecastDay(
                        date=daily_data["time"][i],
                        min_temp=daily_data["temperature_2m_min"][i],
                        max_temp=daily_data["temperature_2m_max"][i],
                        rainfall_mm=daily_data["rain_sum"][i],
                        humidity=avg_humidity,
                        conditions=get_weather_condition(daily_data["weather_code"][i])
                    ))
                
                return WeatherForecastResponse(
                    location=location_name,
                    current=current,
                    forecast=forecast
                )
                
        except Exception as e:
            logger.error(f"Weather forecast error: {e}")
            raise
    
    async def get_historical_weather(
        location: str,
        start_date: str,
        end_date: str,
        metrics: Optional[List[str]] = None
    ) -> HistoricalWeatherResponse:
        """
        Retrieve past weather patterns for seasonal planning.
        Uses Open-Meteo Historical Weather API (free, data from 1940+)
        
        Args:
            location: Location name or coordinates
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            metrics: Specific metrics to retrieve ["temperature", "rainfall", "humidity"]
        """
        try:
            if metrics is None:
                metrics = ["temperature", "rainfall", "humidity"]
            
            lat, lon, location_name = await geocode_location(location)
            
            logger.info(f"Fetching historical weather for: {location_name} ({lat}, {lon})")
            logger.info(f"Period: {start_date} to {end_date}")
            
            # Build daily variables list based on requested metrics
            daily_vars = []
            if "temperature" in metrics:
                daily_vars.append("temperature_2m_mean")
            if "rainfall" in metrics:
                daily_vars.append("rain_sum")
            if "humidity" in metrics:
                daily_vars.append("relative_humidity_2m_mean")
            
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.get(
                    config.archive_url,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": start_date,
                        "end_date": end_date,
                        "daily": ",".join(daily_vars),
                        "timezone": "auto"
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Process historical data
                daily_data = data["daily"]
                data_points = []
                
                temp_sum, rain_sum, humid_sum = 0.0, 0.0, 0.0
                temp_count, rain_count, humid_count = 0, 0, 0
                
                for i in range(len(daily_data["time"])):
                    point_data = {"date": daily_data["time"][i]}
                    
                    if "temperature" in metrics and "temperature_2m_mean" in daily_data:
                        temp = daily_data["temperature_2m_mean"][i]
                        if temp is not None:
                            point_data["temperature"] = temp
                            temp_sum += temp
                            temp_count += 1
                    
                    if "rainfall" in metrics and "rain_sum" in daily_data:
                        rain = daily_data["rain_sum"][i]
                        if rain is not None:
                            point_data["rainfall"] = rain
                            rain_sum += rain
                            rain_count += 1
                    
                    if "humidity" in metrics and "relative_humidity_2m_mean" in daily_data:
                        humid = daily_data["relative_humidity_2m_mean"][i]
                        if humid is not None:
                            point_data["humidity"] = humid
                            humid_sum += humid
                            humid_count += 1
                    
                    data_points.append(HistoricalDataPoint(**point_data))
                
                summary = HistoricalSummary(
                    avg_temperature=temp_sum / temp_count if temp_count > 0 else None,
                    total_rainfall=rain_sum if rain_count > 0 else None,
                    avg_humidity=humid_sum / humid_count if humid_count > 0 else None
                )
                
                return HistoricalWeatherResponse(
                    location=location_name,
                    period=f"{start_date} to {end_date}",
                    data=data_points,
                    summary=summary
                )
                
        except Exception as e:
            logger.error(f"Historical weather error: {e}")
            raise
    
    async def get_climate_alerts(
        location: str,
        alert_types: Optional[List[str]] = None
    ) -> ClimateAlertsResponse:
        """
        Fetch extreme weather warnings based on forecast data.
        Analyzes upcoming weather to generate alerts for agricultural planning.
        
        Args:
            location: Location name or coordinates
            alert_types: Filter by alert types ["storm", "drought", "flood", "frost", "heatwave"]
        """
        try:
            if alert_types is None:
                alert_types = ["storm", "drought", "flood", "frost", "heatwave"]
            
            lat, lon, location_name = await geocode_location(location)
            
            logger.info(f"Analyzing climate alerts for: {location_name} ({lat}, {lon})")
            
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                # Get extended forecast for alert analysis
                response = await client.get(
                    config.forecast_url,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum", "wind_speed_10m_max", "weather_code"],
                        "timezone": "auto",
                        "forecast_days": 14
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                daily = data["daily"]
                active_alerts = []
                
                # Analyze for various alert types
                temps = daily["temperature_2m_max"]
                mins = daily["temperature_2m_min"]
                rainfall = daily["rain_sum"]
                winds = daily["wind_speed_10m_max"]
                weather_codes = daily["weather_code"]
                
                # Check for heatwave
                if "heatwave" in alert_types:
                    high_temp_days = sum(1 for t in temps[:7] if t > 35)
                    if high_temp_days >= 3:
                        active_alerts.append(ClimateAlert(
                            type="heatwave",
                            severity="high",
                            title="Heatwave Warning",
                            description=f"Temperatures exceeding 35°C expected for {high_temp_days} days in the coming week",
                            start_time=datetime.now().isoformat(),
                            end_time=(datetime.now() + timedelta(days=7)).isoformat(),
                            recommendations=[
                                "Increase irrigation frequency",
                                "Provide shade for sensitive crops",
                                "Monitor soil moisture levels closely",
                                "Consider delaying planting of heat-sensitive crops"
                            ]
                        ))
                
                # Check for frost
                if "frost" in alert_types:
                    frost_days = [(i, t) for i, t in enumerate(mins[:7]) if t < 0]
                    if frost_days:
                        active_alerts.append(ClimateAlert(
                            type="frost",
                            severity="high",
                            title="Frost Alert",
                            description=f"Freezing temperatures expected on {len(frost_days)} day(s)",
                            start_time=datetime.now().isoformat(),
                            end_time=(datetime.now() + timedelta(days=7)).isoformat(),
                            recommendations=[
                                "Protect sensitive plants with covers",
                                "Harvest mature crops before frost",
                                "Delay planting frost-sensitive crops",
                                "Use frost protection methods (sprinklers, heaters)"
                            ]
                        ))
                
                # Check for drought conditions
                if "drought" in alert_types:
                    total_rain = sum(rainfall[:14])
                    if total_rain < 10:  # Less than 10mm in 2 weeks
                        active_alerts.append(ClimateAlert(
                            type="drought",
                            severity="moderate",
                            title="Dry Period Advisory",
                            description=f"Low rainfall expected - only {total_rain:.1f}mm in the next 2 weeks",
                            start_time=datetime.now().isoformat(),
                            end_time=(datetime.now() + timedelta(days=14)).isoformat(),
                            recommendations=[
                                "Implement water conservation measures",
                                "Consider drought-resistant crop varieties",
                                "Monitor soil moisture levels",
                                "Plan irrigation schedule carefully"
                            ]
                        ))
                
                # Check for heavy rain/flood risk
                if "flood" in alert_types or "storm" in alert_types:
                    heavy_rain_days = [(i, r) for i, r in enumerate(rainfall[:7]) if r > 50]
                    if heavy_rain_days:
                        alert_type = "flood" if "flood" in alert_types else "storm"
                        active_alerts.append(ClimateAlert(
                            type=alert_type,
                            severity="high",
                            title="Heavy Rainfall Warning",
                            description=f"Intense rainfall expected - up to {max(r for _, r in heavy_rain_days):.1f}mm in a single day",
                            start_time=datetime.now().isoformat(),
                            end_time=(datetime.now() + timedelta(days=7)).isoformat(),
                            recommendations=[
                                "Ensure proper drainage systems are clear",
                                "Delay field operations during heavy rain",
                                "Protect young plants from waterlogging",
                                "Monitor for soil erosion",
                                "Harvest mature crops if possible"
                            ]
                        ))
                
                # Check for storms (high winds + rain)
                if "storm" in alert_types:
                    storm_days = [(i, w) for i, w in enumerate(winds[:7]) if w > 60]  # >60 km/h
                    if storm_days:
                        active_alerts.append(ClimateAlert(
                            type="storm",
                            severity="high",
                            title="Strong Wind Warning",
                            description=f"High winds expected - up to {max(w for _, w in storm_days):.1f} km/h",
                            start_time=datetime.now().isoformat(),
                            end_time=(datetime.now() + timedelta(days=7)).isoformat(),
                            recommendations=[
                                "Secure loose equipment and structures",
                                "Stake or support tall plants",
                                "Delay spraying operations",
                                "Check greenhouses and protective structures"
                            ]
                        ))
                
                return ClimateAlertsResponse(
                    location=location_name,
                    active_alerts=active_alerts
                )
                
        except Exception as e:
            logger.error(f"Climate alerts error: {e}")
            raise
    
    # Create a unified handler that routes to the appropriate function
    async def weather_climate_handler(input_message: str) -> Dict[str, Any]:
        """
        Main handler for weather climate requests.
        Expects input in format: {"function": "function_name", "params": {...}}
        """
        try:
            import json
            request = json.loads(input_message) if isinstance(input_message, str) else input_message
            
            function_name = request.get("function")
            params = request.get("params", {})
            
            if function_name == "get_weather_forecast":
                result = await get_weather_forecast(**params)
                return result.model_dump()
            elif function_name == "get_historical_weather":
                result = await get_historical_weather(**params)
                return result.model_dump()
            elif function_name == "get_climate_alerts":
                result = await get_climate_alerts(**params)
                return result.model_dump()
            else:
                return {
                    "error": f"Unknown function: {function_name}",
                    "available_functions": [
                        "get_weather_forecast",
                        "get_historical_weather", 
                        "get_climate_alerts"
                    ]
                }
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e)}
    
    try:
        yield FunctionInfo.create(single_fn=weather_climate_handler)
    except GeneratorExit:
        logger.warning("Weather climate function exited early!")
    finally:
        logger.info("Cleaning up weather_climate workflow.")