import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import httpx
from pydantic import Field, BaseModel

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class WeatherClimateFunctionConfig(FunctionBaseConfig, name="weather_climate"):
    """
    AI-Enhanced Weather and Climate data retrieval function using Open-Meteo API.
    Provides weather forecasts, historical weather patterns, climate alerts with AI-powered analysis.
    """
    forecast_url: str = Field(default="https://api.open-meteo.com/v1/forecast", description="Forecast API URL")
    archive_url: str = Field(default="https://archive-api.open-meteo.com/v1/archive", description="Historical Archive API URL")
    geocoding_url: str = Field(default="https://geocoding-api.open-meteo.com/v1", description="Geocoding API URL")
    timeout: int = Field(default=30, description="API request timeout in seconds")
    llm_name: str = Field(description="Name of the LLM to use for AI analysis")
    max_history: int = Field(default=15, description="Maximum conversation history")


# Response models (keeping all existing models)
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


@register_function(config_type=WeatherClimateFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def weather_climate_function(
    config: WeatherClimateFunctionConfig, builder: Builder
):
    """
    AI-Enhanced Weather and Climate tool using Open-Meteo API with intelligent analysis
    """
    
    # Initialize LLM for AI analysis
    llm_ref = LLMRef(config.llm_name)
    llm = await builder.get_llm(llm_ref, LLMFrameworkEnum.LANGCHAIN)
    
    # Create AI analysis prompt
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Agricultural Weather Advisor and Climate Analyst.

Your role is to analyze weather and climate data and provide:
- Practical agricultural insights and recommendations
- Risk assessments for crops and farming operations
- Strategic planning advice based on weather patterns
- Actionable steps farmers can take to optimize their operations

When analyzing weather data:
1. Focus on agricultural impacts and opportunities
2. Provide specific, actionable recommendations
3. Highlight risks and how to mitigate them
4. Consider seasonal patterns and climate trends
5. Be clear about timing and urgency of actions

Be professional, practical, and focused on helping farmers make data-driven decisions.
"""),
        ("human", "{input}")
    ])
    
    # Create analysis chain
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    
    # Conversation history
    conversation_history = []
    
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
            0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "foggy", 48: "depositing rime fog",
            51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
            61: "slight rain", 63: "moderate rain", 65: "heavy rain",
            71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
            80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
            85: "slight snow showers", 86: "heavy snow showers",
            95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail"
        }
        return conditions.get(weather_code, "unknown")
    
    def format_weather_data(data: WeatherForecastResponse) -> str:
        """Format weather forecast data for AI analysis"""
        result = f"📍 **LOCATION**: {data.location}\n\n"
        
        result += f"🌡️ **CURRENT CONDITIONS**:\n"
        result += f"• Temperature: {data.current.temperature}°C\n"
        result += f"• Humidity: {data.current.humidity}%\n"
        result += f"• Rainfall: {data.current.rainfall}mm\n"
        result += f"• Wind Speed: {data.current.wind_speed} km/h\n"
        result += f"• Conditions: {data.current.conditions}\n\n"
        
        result += f"📅 **{len(data.forecast)}-DAY FORECAST**:\n"
        for day in data.forecast[:7]:  # Show first 7 days in detail
            result += f"\n**{day.date}** ({day.conditions}):\n"
            result += f"  • Temp: {day.min_temp}°C - {day.max_temp}°C\n"
            result += f"  • Rainfall: {day.rainfall_mm}mm\n"
            result += f"  • Humidity: {day.humidity}%\n"
        
        return result
    
    def format_historical_data(data: HistoricalWeatherResponse) -> str:
        """Format historical weather data for AI analysis"""
        result = f"📍 **LOCATION**: {data.location}\n"
        result += f"📆 **PERIOD**: {data.period}\n\n"
        
        result += f"📊 **SUMMARY STATISTICS**:\n"
        if data.summary.avg_temperature:
            result += f"• Average Temperature: {data.summary.avg_temperature:.1f}°C\n"
        if data.summary.total_rainfall:
            result += f"• Total Rainfall: {data.summary.total_rainfall:.1f}mm\n"
        if data.summary.avg_humidity:
            result += f"• Average Humidity: {data.summary.avg_humidity:.1f}%\n"
        
        result += f"\n📈 **DAILY DATA** ({len(data.data)} days recorded):\n"
        # Show sample of data points
        for point in data.data[:5]:
            result += f"\n**{point.date}**:\n"
            if point.temperature:
                result += f"  • Temperature: {point.temperature:.1f}°C\n"
            if point.rainfall:
                result += f"  • Rainfall: {point.rainfall:.1f}mm\n"
            if point.humidity:
                result += f"  • Humidity: {point.humidity:.1f}%\n"
        
        if len(data.data) > 5:
            result += f"\n... and {len(data.data) - 5} more days of data\n"
        
        return result
    
    def format_alerts_data(data: ClimateAlertsResponse) -> str:
        """Format climate alerts data for AI analysis"""
        result = f"📍 **LOCATION**: {data.location}\n"
        result += f"⚠️ **ACTIVE ALERTS**: {len(data.active_alerts)}\n\n"
        
        if not data.active_alerts:
            result += "✅ No active weather alerts at this time.\n"
        else:
            for i, alert in enumerate(data.active_alerts, 1):
                severity_emoji = "🔴" if alert.severity == "high" else "🟡"
                result += f"{severity_emoji} **ALERT #{i}: {alert.title}**\n"
                result += f"• Type: {alert.type.upper()}\n"
                result += f"• Severity: {alert.severity.upper()}\n"
                result += f"• Description: {alert.description}\n"
                result += f"• Timeframe: {alert.start_time[:10]} to {alert.end_time[:10]}\n"
                result += f"• Recommendations:\n"
                for rec in alert.recommendations:
                    result += f"  - {rec}\n"
                result += "\n"
        
        return result
    
    async def get_weather_forecast(
        location: str,
        days: int = 7,
        units: str = "celsius"
    ) -> WeatherForecastResponse:
        """Get current and upcoming weather data for a specific location."""
        try:
            days = min(days, 16)
            lat, lon, location_name = await geocode_location(location)
            
            temp_unit = "celsius" if units == "celsius" else "fahrenheit"
            
            logger.info(f"Fetching weather forecast for: {location_name} ({lat}, {lon})")
            
            async with httpx.AsyncClient(timeout=config.timeout) as client:
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
                
                current_data = data["current"]
                current = CurrentWeather(
                    temperature=current_data["temperature_2m"],
                    humidity=current_data["relative_humidity_2m"],
                    rainfall=current_data.get("rain", 0.0),
                    wind_speed=current_data["wind_speed_10m"],
                    conditions=get_weather_condition(current_data["weather_code"])
                )
                
                daily_data = data["daily"]
                forecast = []
                for i in range(len(daily_data["time"])):
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
        """Retrieve past weather patterns for seasonal planning."""
        try:
            if metrics is None:
                metrics = ["temperature", "rainfall", "humidity"]
            
            lat, lon, location_name = await geocode_location(location)
            
            logger.info(f"Fetching historical weather for: {location_name}")
            
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
        """Fetch extreme weather warnings based on forecast data."""
        try:
            if alert_types is None:
                alert_types = ["storm", "drought", "flood", "frost", "heatwave"]
            
            lat, lon, location_name = await geocode_location(location)
            
            logger.info(f"Analyzing climate alerts for: {location_name}")
            
            async with httpx.AsyncClient(timeout=config.timeout) as client:
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
                
                temps = daily["temperature_2m_max"]
                mins = daily["temperature_2m_min"]
                rainfall = daily["rain_sum"]
                winds = daily["wind_speed_10m_max"]
                
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
                
                # Check for drought
                if "drought" in alert_types:
                    total_rain = sum(rainfall[:14])
                    if total_rain < 10:
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
                
                # Check for heavy rain/flood
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
                
                # Check for storms
                if "storm" in alert_types:
                    storm_days = [(i, w) for i, w in enumerate(winds[:7]) if w > 60]
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
    
    # Main handler with AI analysis
    async def weather_climate_handler(input_message: str) -> str:
        """Main handler with AI-powered analysis"""
        nonlocal conversation_history
        
        try:
            import json
            
            # Parse the input_message string to get the actual request
            logger.info(f"Received input_message: {input_message[:100]}...")
            request = json.loads(input_message)
            
            # Add to conversation history
            conversation_history.append({
                "role": "user",
                "content": input_message,
                "timestamp": datetime.now().isoformat()
            })
            
            function_name = request.get("function")
            params = request.get("params", {})
            user_query = request.get("query", "")  # Optional natural language query
            
            # Fetch weather data
            weather_data_formatted = ""
            
            if function_name == "get_weather_forecast":
                logger.info(f"Fetching weather forecast with AI analysis...")
                result = await get_weather_forecast(**params)
                weather_data_formatted = format_weather_data(result)
                
            elif function_name == "get_historical_weather":
                logger.info(f"Fetching historical weather with AI analysis...")
                result = await get_historical_weather(**params)
                weather_data_formatted = format_historical_data(result)
                
            elif function_name == "get_climate_alerts":
                logger.info(f"Fetching climate alerts with AI analysis...")
                result = await get_climate_alerts(**params)
                weather_data_formatted = format_alerts_data(result)
                
            else:
                return json.dumps({
                    "error": f"Unknown function: {function_name}",
                    "available_functions": [
                        "get_weather_forecast",
                        "get_historical_weather",
                        "get_climate_alerts"
                    ]
                })
            
            # Prepare context for AI analysis
            analysis_context = f"""
User Request: {user_query if user_query else f"Analysis of {function_name}"}

Here is the weather data to analyze:

{weather_data_formatted}

Please provide:
1. A clear summary of the key weather insights
2. Agricultural implications and opportunities
3. Specific recommendations for farmers
4. Any risks or concerns that need attention
5. Optimal timing for farming activities based on this data

Focus on practical, actionable advice that farmers can implement immediately.
"""
            
            # Add conversation context
            if len(conversation_history) > 2:
                analysis_context += f"\n\nThis is exchange #{len(conversation_history)//2 + 1} in our conversation."
            
            # Get AI analysis
            logger.info("Generating AI analysis...")
            ai_analysis = await analysis_chain.ainvoke({"input": analysis_context})
            
            # Combine weather data and AI analysis
            final_response = f"""
{weather_data_formatted}

{'='*60}
🤖 **AI AGRICULTURAL ANALYSIS**
{'='*60}

{ai_analysis}

---
📊 **Data Source**: Open-Meteo API
🔄 **Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
💬 **Session**: {len(conversation_history)//2} exchanges completed
"""
            
            # Add to conversation history
            conversation_history.append({
                "role": "assistant",
                "content": final_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Trim history
            if len(conversation_history) > config.max_history * 2:
                conversation_history = conversation_history[-(config.max_history * 2):]
            
            return final_response
            
        except Exception as e:
            logger.error(f"Weather climate handler error: {e}")
            return json.dumps({"error": str(e)})
    
    try:
        yield FunctionInfo.create(single_fn=weather_climate_handler)
    except GeneratorExit:
        logger.warning("Weather climate function exited early!")
    finally:
        logger.info("Cleaning up weather_climate workflow.")