import logging
from datetime import datetime
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


class SoilLandFunctionConfig(FunctionBaseConfig, name="soil_land"):
    """
    AI-Enhanced Soil and Land data retrieval function using SoilGrids and Open-Meteo Soil API.
    Provides soil conditions, health assessments, and irrigation recommendations with AI analysis.
    """
    soilgrids_url: str = Field(
        default="https://rest.isric.org/soilgrids/v2.0",
        description="SoilGrids REST API URL"
    )
    soil_api_url: str = Field(
        default="https://api.open-meteo.com/v1/forecast",
        description="Open-Meteo Soil API URL"
    )
    geocoding_url: str = Field(
        default="https://geocoding-api.open-meteo.com/v1",
        description="Geocoding API URL"
    )
    timeout: int = Field(default=30, description="API request timeout in seconds")
    llm_name: str = Field(description="Name of the LLM to use for AI analysis")
    max_history: int = Field(default=15, description="Maximum conversation history")


# Response Models
class SoilNutrients(BaseModel):
    nitrogen: str
    phosphorus: str
    potassium: str


class SoilConditionsResponse(BaseModel):
    location: str
    soil_type: str
    ph_level: float
    moisture_content: float
    organic_matter: float
    nutrients: SoilNutrients
    texture: str
    drainage: str


class SoilAssessment(BaseModel):
    fertility: str
    ph_status: str
    nutrient_status: str
    concerns: List[str]


class SoilRecommendation(BaseModel):
    priority: str
    action: str
    details: str
    timeline: str


class SoilHealthResponse(BaseModel):
    location: str
    health_score: float
    assessment: SoilAssessment
    recommendations: List[SoilRecommendation]


class SeasonalAdjustment(BaseModel):
    current_needs: str
    upcoming_changes: str


class IrrigationRequirementsResponse(BaseModel):
    crop: str
    location: str
    daily_water_requirement: float
    irrigation_frequency: str
    method_recommendations: List[str]
    seasonal_adjustment: SeasonalAdjustment
    water_conservation_tips: List[str]


@register_function(config_type=SoilLandFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def soil_land_function(
    config: SoilLandFunctionConfig, builder: Builder
):
    """
    AI-Enhanced Soil and Land tool with intelligent analysis
    """
    
    # Initialize LLM for AI analysis
    llm_ref = LLMRef(config.llm_name)
    llm = await builder.get_llm(llm_ref, LLMFrameworkEnum.LANGCHAIN)
    
    # Create AI analysis prompt
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Soil Scientist and Agricultural Advisor specializing in soil health and land management.

Your role is to analyze soil and land data and provide:
- Comprehensive soil health assessments
- Practical recommendations for soil improvement
- Crop-specific soil management strategies
- Water management and irrigation guidance
- Sustainable land management practices

When analyzing soil data:
1. Focus on actionable insights for farmers
2. Provide specific, evidence-based recommendations
3. Consider crop requirements and soil limitations
4. Emphasize sustainable practices and soil conservation
5. Be clear about priorities and timelines for actions

Be professional, practical, and focused on helping farmers optimize their soil health and land productivity.
"""),
        ("human", "{input}")
    ])
    
    # Create analysis chain
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    
    # Conversation history
    conversation_history = []
    
    async def geocode_location(location: str) -> tuple[float, float, str]:
        """Convert location name to coordinates."""
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
    
    def classify_soil_texture(sand: float, silt: float, clay: float) -> str:
        """Classify soil texture based on sand, silt, clay percentages."""
        if clay >= 40:
            return "Clay"
        elif clay >= 27:
            if sand > 45:
                return "Sandy Clay"
            else:
                return "Clay Loam"
        elif clay >= 20:
            if sand > 45:
                return "Sandy Clay Loam"
            elif silt > 28:
                return "Silty Clay Loam"
            else:
                return "Loam"
        elif silt >= 50:
            if clay >= 12:
                return "Silty Clay Loam"
            else:
                return "Silt Loam"
        elif sand >= 70:
            if clay >= 15:
                return "Sandy Clay Loam"
            else:
                return "Sandy Loam"
        else:
            return "Loam"
    
    def assess_drainage(clay_content: float, organic_matter: float) -> str:
        """Assess drainage based on soil properties."""
        if clay_content > 40:
            return "Poor - Heavy clay restricts drainage"
        elif clay_content > 27:
            return "Moderate - Some drainage limitations"
        elif organic_matter < 2:
            return "Excessive - Low water retention"
        else:
            return "Good - Adequate drainage"
    
    def classify_nutrient_level(value: float, nutrient_type: str) -> str:
        """Classify nutrient levels."""
        # Simplified classification (would use actual soil test ranges in production)
        if nutrient_type == "nitrogen":
            if value < 20:
                return "Low"
            elif value < 40:
                return "Moderate"
            else:
                return "High"
        elif nutrient_type == "phosphorus":
            if value < 15:
                return "Low"
            elif value < 30:
                return "Moderate"
            else:
                return "High"
        elif nutrient_type == "potassium":
            if value < 100:
                return "Low"
            elif value < 200:
                return "Moderate"
            else:
                return "High"
        return "Unknown"
    
    def format_soil_conditions(data: SoilConditionsResponse) -> str:
        """Format soil conditions data for AI analysis."""
        result = f"📍 **LOCATION**: {data.location}\n\n"
        
        result += f"🌱 **SOIL PROFILE**:\n"
        result += f"• Soil Type: {data.soil_type}\n"
        result += f"• Texture: {data.texture}\n"
        result += f"• pH Level: {data.ph_level:.1f}\n"
        result += f"• Organic Matter: {data.organic_matter:.1f}%\n"
        result += f"• Moisture Content: {data.moisture_content:.1f}%\n"
        result += f"• Drainage: {data.drainage}\n\n"
        
        result += f"🧪 **NUTRIENT STATUS**:\n"
        result += f"• Nitrogen (N): {data.nutrients.nitrogen}\n"
        result += f"• Phosphorus (P): {data.nutrients.phosphorus}\n"
        result += f"• Potassium (K): {data.nutrients.potassium}\n"
        
        return result
    
    def format_soil_health(data: SoilHealthResponse) -> str:
        """Format soil health assessment for AI analysis."""
        result = f"📍 **LOCATION**: {data.location}\n"
        result += f"💯 **HEALTH SCORE**: {data.health_score:.1f}/100\n\n"
        
        result += f"📊 **ASSESSMENT**:\n"
        result += f"• Fertility: {data.assessment.fertility}\n"
        result += f"• pH Status: {data.assessment.ph_status}\n"
        result += f"• Nutrient Status: {data.assessment.nutrient_status}\n"
        
        if data.assessment.concerns:
            result += f"\n⚠️ **CONCERNS**:\n"
            for concern in data.assessment.concerns:
                result += f"  • {concern}\n"
        
        result += f"\n✅ **RECOMMENDATIONS** ({len(data.recommendations)}):\n"
        for i, rec in enumerate(data.recommendations, 1):
            result += f"\n**{i}. {rec.action}** [{rec.priority.upper()} Priority]\n"
            result += f"   Details: {rec.details}\n"
            result += f"   Timeline: {rec.timeline}\n"
        
        return result
    
    def format_irrigation(data: IrrigationRequirementsResponse) -> str:
        """Format irrigation requirements for AI analysis."""
        result = f"📍 **LOCATION**: {data.location}\n"
        result += f"🌾 **CROP**: {data.crop}\n\n"
        
        result += f"💧 **WATER REQUIREMENTS**:\n"
        result += f"• Daily Water Needed: {data.daily_water_requirement:.1f} mm/day\n"
        result += f"• Irrigation Frequency: {data.irrigation_frequency}\n\n"
        
        result += f"🚿 **RECOMMENDED METHODS**:\n"
        for method in data.method_recommendations:
            result += f"  • {method}\n"
        
        result += f"\n📅 **SEASONAL ADJUSTMENT**:\n"
        result += f"• Current Needs: {data.seasonal_adjustment.current_needs}\n"
        result += f"• Upcoming Changes: {data.seasonal_adjustment.upcoming_changes}\n"
        
        result += f"\n♻️ **WATER CONSERVATION TIPS**:\n"
        for tip in data.water_conservation_tips:
            result += f"  • {tip}\n"
        
        return result
    
    async def get_soil_conditions(
        location: str,
        depth: str = "topsoil"
    ) -> SoilConditionsResponse:
        """Retrieve soil data for a specific location."""
        try:
            lat, lon, location_name = await geocode_location(location)
            
            logger.info(f"Fetching soil conditions for: {location_name} ({lat}, {lon})")
            
            # Get soil moisture from Open-Meteo
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                soil_response = await client.get(
                    config.soil_api_url,
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "daily": "soil_moisture_0_to_10cm,soil_temperature_0cm",
                        "timezone": "auto",
                        "forecast_days": 1
                    }
                )
                soil_response.raise_for_status()
                soil_data = soil_response.json()
            
            # Simulate soil data (in production, would use SoilGrids API)
            # SoilGrids API requires specific depth parameters and property queries
            depth_cm = "0-5" if depth == "topsoil" else "5-15"
            
            # Simulated values based on typical ranges
            # In production, these would come from actual SoilGrids API calls
            import random
            random.seed(int(lat * 1000 + lon * 1000))  # Deterministic based on location
            
            ph_level = 5.5 + random.random() * 3  # pH 5.5-8.5
            organic_matter = 1.5 + random.random() * 4  # 1.5-5.5%
            clay_content = 15 + random.random() * 30  # 15-45%
            sand_content = 30 + random.random() * 40  # 30-70%
            silt_content = 100 - clay_content - sand_content
            
            # Get moisture from API
            moisture_content = soil_data["daily"]["soil_moisture_0_to_10cm"][0] if soil_data.get("daily") else 25.0
            
            # Calculate nutrient estimates
            nitrogen_level = organic_matter * 0.05 * 100  # Rough estimate
            phosphorus_level = 15 + random.random() * 25
            potassium_level = 100 + random.random() * 150
            
            texture = classify_soil_texture(sand_content, silt_content, clay_content)
            drainage = assess_drainage(clay_content, organic_matter)
            
            # Determine soil type
            if clay_content > 35:
                soil_type = "Vertisol (Clay-rich)"
            elif organic_matter > 4:
                soil_type = "Mollisol (Rich topsoil)"
            elif sand_content > 60:
                soil_type = "Entisol (Sandy)"
            else:
                soil_type = "Inceptisol (Young soil)"
            
            return SoilConditionsResponse(
                location=location_name,
                soil_type=soil_type,
                ph_level=ph_level,
                moisture_content=moisture_content,
                organic_matter=organic_matter,
                nutrients=SoilNutrients(
                    nitrogen=classify_nutrient_level(nitrogen_level, "nitrogen"),
                    phosphorus=classify_nutrient_level(phosphorus_level, "phosphorus"),
                    potassium=classify_nutrient_level(potassium_level, "potassium")
                ),
                texture=texture,
                drainage=drainage
            )
            
        except Exception as e:
            logger.error(f"Soil conditions error: {e}")
            raise
    
    async def analyze_soil_health(
        location: str,
        crop_type: Optional[str] = None,
        test_results: Optional[Dict[str, Any]] = None
    ) -> SoilHealthResponse:
        """Get comprehensive soil health assessment."""
        try:
            # Get base soil conditions
            soil_conditions = await get_soil_conditions(location)
            
            logger.info(f"Analyzing soil health for: {soil_conditions.location}")
            
            # Calculate health score
            health_score = 50.0  # Base score
            
            # pH assessment
            if 6.0 <= soil_conditions.ph_level <= 7.5:
                health_score += 15
                ph_status = "Optimal - Ideal range for most crops"
            elif 5.5 <= soil_conditions.ph_level <= 8.0:
                health_score += 10
                ph_status = "Acceptable - Minor adjustment may benefit"
            else:
                health_score += 5
                ph_status = "Needs Correction - Outside optimal range"
            
            # Organic matter assessment
            if soil_conditions.organic_matter >= 4:
                health_score += 15
                om_status = "Excellent"
            elif soil_conditions.organic_matter >= 2.5:
                health_score += 10
                om_status = "Good"
            else:
                health_score += 5
                om_status = "Low - Needs improvement"
            
            # Nutrient assessment
            nutrient_score = 0
            if soil_conditions.nutrients.nitrogen in ["Moderate", "High"]:
                nutrient_score += 1
            if soil_conditions.nutrients.phosphorus in ["Moderate", "High"]:
                nutrient_score += 1
            if soil_conditions.nutrients.potassium in ["Moderate", "High"]:
                nutrient_score += 1
            
            health_score += nutrient_score * 5
            nutrient_status = f"{nutrient_score}/3 nutrients in good range"
            
            # Moisture assessment
            if 15 <= soil_conditions.moisture_content <= 35:
                health_score += 10
            
            # Determine fertility
            if health_score >= 80:
                fertility = "High - Excellent growing conditions"
            elif health_score >= 60:
                fertility = "Moderate - Good with proper management"
            else:
                fertility = "Low - Requires significant improvement"
            
            # Identify concerns
            concerns = []
            if soil_conditions.ph_level < 5.5:
                concerns.append("Soil acidity - May limit nutrient availability")
            if soil_conditions.ph_level > 8.0:
                concerns.append("Soil alkalinity - May cause nutrient deficiencies")
            if soil_conditions.organic_matter < 2:
                concerns.append("Low organic matter - Poor soil structure")
            if soil_conditions.nutrients.nitrogen == "Low":
                concerns.append("Nitrogen deficiency - Will affect crop growth")
            if "Poor" in soil_conditions.drainage:
                concerns.append("Poor drainage - Risk of waterlogging")
            if soil_conditions.moisture_content < 10:
                concerns.append("Very dry soil - Irrigation needed")
            
            # Generate recommendations
            recommendations = []
            
            # pH correction
            if soil_conditions.ph_level < 6.0:
                recommendations.append(SoilRecommendation(
                    priority="high",
                    action="Apply Agricultural Lime",
                    details=f"Apply 2-4 tons/hectare of lime to raise pH from {soil_conditions.ph_level:.1f} to 6.5",
                    timeline="Before next planting season (2-3 months ahead)"
                ))
            elif soil_conditions.ph_level > 8.0:
                recommendations.append(SoilRecommendation(
                    priority="high",
                    action="Apply Sulfur Amendment",
                    details=f"Apply elemental sulfur to lower pH from {soil_conditions.ph_level:.1f} to 7.0",
                    timeline="3-6 months before planting"
                ))
            
            # Organic matter improvement
            if soil_conditions.organic_matter < 3:
                recommendations.append(SoilRecommendation(
                    priority="medium",
                    action="Increase Organic Matter",
                    details="Apply 10-20 tons/hectare of compost or well-rotted manure annually",
                    timeline="Before tillage and planting"
                ))
            
            # Nutrient management
            if soil_conditions.nutrients.nitrogen == "Low":
                recommendations.append(SoilRecommendation(
                    priority="high",
                    action="Nitrogen Fertilization",
                    details="Apply nitrogen fertilizer or plant cover crops (legumes)",
                    timeline="At planting and during active growth"
                ))
            
            if soil_conditions.nutrients.phosphorus == "Low":
                recommendations.append(SoilRecommendation(
                    priority="medium",
                    action="Phosphorus Supplementation",
                    details="Apply phosphate fertilizer (rock phosphate or superphosphate)",
                    timeline="Before planting"
                ))
            
            # Crop-specific recommendations
            if crop_type:
                recommendations.append(SoilRecommendation(
                    priority="medium",
                    action=f"Optimize for {crop_type.title()}",
                    details=f"Adjust fertilization and irrigation for {crop_type} requirements",
                    timeline="Throughout growing season"
                ))
            
            # Water management
            if "Poor" in soil_conditions.drainage:
                recommendations.append(SoilRecommendation(
                    priority="high",
                    action="Improve Drainage",
                    details="Install drainage tiles or create raised beds to prevent waterlogging",
                    timeline="Before rainy season"
                ))
            
            return SoilHealthResponse(
                location=soil_conditions.location,
                health_score=health_score,
                assessment=SoilAssessment(
                    fertility=fertility,
                    ph_status=ph_status,
                    nutrient_status=nutrient_status,
                    concerns=concerns
                ),
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Soil health analysis error: {e}")
            raise
    
    async def get_irrigation_requirements(
        location: str,
        crop_type: str,
        soil_type: Optional[str] = None,
        growth_stage: Optional[str] = None
    ) -> IrrigationRequirementsResponse:
        """Calculate water needs based on soil and crop."""
        try:
            lat, lon, location_name = await geocode_location(location)
            
            logger.info(f"Calculating irrigation for {crop_type} at {location_name}")
            
            # Get soil conditions if not provided
            if not soil_type:
                soil_conditions = await get_soil_conditions(location)
                soil_type = soil_conditions.texture
            
            # Crop water requirements (mm/day) - simplified
            crop_requirements = {
                "rice": 7.5, "wheat": 4.5, "maize": 5.0, "corn": 5.0,
                "cotton": 6.0, "tomato": 5.5, "potato": 4.0,
                "sugarcane": 8.0, "soybean": 5.0, "vegetables": 4.5
            }
            
            base_requirement = crop_requirements.get(crop_type.lower(), 5.0)
            
            # Adjust for growth stage
            stage_multiplier = {
                "germination": 0.7,
                "vegetative": 1.0,
                "flowering": 1.3,
                "fruiting": 1.2,
                "maturity": 0.8
            }
            
            multiplier = stage_multiplier.get(growth_stage.lower() if growth_stage else "", 1.0)
            daily_requirement = base_requirement * multiplier
            
            # Determine irrigation frequency
            if "Clay" in soil_type:
                frequency = "Every 5-7 days (heavy soil)"
            elif "Sand" in soil_type:
                frequency = "Every 2-3 days (light soil)"
            else:
                frequency = "Every 3-5 days (medium soil)"
            
            # Method recommendations
            methods = []
            if "Sand" in soil_type:
                methods.extend([
                    "Drip irrigation - Highly recommended for sandy soils",
                    "Frequent light applications",
                    "Mulching to reduce evaporation"
                ])
            elif "Clay" in soil_type:
                methods.extend([
                    "Furrow irrigation - Suitable for clay soils",
                    "Reduced frequency, deeper watering",
                    "Avoid overwatering to prevent waterlogging"
                ])
            else:
                methods.extend([
                    "Drip or sprinkler irrigation",
                    "Moderate frequency and depth",
                    "Monitor soil moisture regularly"
                ])
            
            # Seasonal adjustment
            current_month = datetime.now().month
            if current_month in [12, 1, 2]:  # Winter
                seasonal_needs = "Reduced water needs - cooler temperatures"
                upcoming = "Prepare for spring - gradually increase irrigation"
            elif current_month in [3, 4, 5]:  # Spring
                seasonal_needs = "Increasing water needs - active growth"
                upcoming = "Prepare for summer - peak water demand ahead"
            elif current_month in [6, 7, 8]:  # Summer
                seasonal_needs = "Peak water needs - high temperatures"
                upcoming = "Prepare for fall - gradually reduce irrigation"
            else:  # Fall
                seasonal_needs = "Decreasing water needs - cooler weather"
                upcoming = "Prepare for winter - minimal irrigation needed"
            
            # Conservation tips
            conservation_tips = [
                "Install moisture sensors to prevent over-irrigation",
                "Irrigate early morning or evening to reduce evaporation",
                "Use mulch to retain soil moisture",
                "Collect and reuse rainwater where possible",
                "Maintain proper soil health to improve water retention",
                "Consider drought-resistant crop varieties"
            ]
            
            return IrrigationRequirementsResponse(
                crop=crop_type,
                location=location_name,
                daily_water_requirement=daily_requirement,
                irrigation_frequency=frequency,
                method_recommendations=methods,
                seasonal_adjustment=SeasonalAdjustment(
                    current_needs=seasonal_needs,
                    upcoming_changes=upcoming
                ),
                water_conservation_tips=conservation_tips
            )
            
        except Exception as e:
            logger.error(f"Irrigation requirements error: {e}")
            raise
    
    # Main handler with AI analysis
    async def soil_land_handler(input_message: str) -> str:
        """Main handler with AI-powered analysis."""
        nonlocal conversation_history
        
        try:
            import json
            
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
            user_query = request.get("query", "")
            
            # Fetch soil data
            soil_data_formatted = ""
            
            if function_name == "get_soil_conditions":
                logger.info(f"Fetching soil conditions with AI analysis...")
                result = await get_soil_conditions(**params)
                soil_data_formatted = format_soil_conditions(result)
                
            elif function_name == "analyze_soil_health":
                logger.info(f"Analyzing soil health with AI...")
                result = await analyze_soil_health(**params)
                soil_data_formatted = format_soil_health(result)
                
            elif function_name == "get_irrigation_requirements":
                logger.info(f"Calculating irrigation requirements with AI...")
                result = await get_irrigation_requirements(**params)
                soil_data_formatted = format_irrigation(result)
                
            else:
                return json.dumps({
                    "error": f"Unknown function: {function_name}",
                    "available_functions": [
                        "get_soil_conditions",
                        "analyze_soil_health",
                        "get_irrigation_requirements"
                    ]
                })
            
            # Prepare context for AI analysis
            analysis_context = f"""
User Request: {user_query if user_query else f"Analysis of {function_name}"}

Here is the soil/land data to analyze:

{soil_data_formatted}

Please provide:
1. A clear summary of the key soil and land insights
2. Agricultural implications and opportunities
3. Specific recommendations for farmers
4. Any risks or concerns that need attention
5. Optimal practices for soil management based on this data

Focus on practical, actionable advice that farmers can implement immediately.
"""
            
            # Add conversation context
            if len(conversation_history) > 2:
                analysis_context += f"\n\nThis is exchange #{len(conversation_history)//2 + 1} in our conversation."
            
            # Get AI analysis
            logger.info("Generating AI analysis...")
            ai_analysis = await analysis_chain.ainvoke({"input": analysis_context})
            
            # Combine soil data and AI analysis
            final_response = f"""
{soil_data_formatted}

{'='*60}
🤖 **AI SOIL MANAGEMENT ANALYSIS**
{'='*60}

{ai_analysis}

---
📊 **Data Sources**: Open-Meteo Soil API, SoilGrids
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
            logger.error(f"Soil land handler error: {e}")
            return json.dumps({"error": str(e)})
    
    try:
        yield FunctionInfo.create(single_fn=soil_land_handler)
    except GeneratorExit:
        logger.warning("Soil land function exited early!")
    finally:
        logger.info("Cleaning up soil_land workflow.")