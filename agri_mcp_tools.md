# Agricultural Advisory System - MCP Tools Documentation

## Weather & Climate Tools

### 1. get_weather_forecast
Get current and upcoming weather data for a specific location.

**Parameters:**
- `location` (string, required): Location name or coordinates (lat,lon)
- `days` (integer, optional): Number of days to forecast (default: 7, max: 14)
- `units` (string, optional): Temperature units - "celsius" or "fahrenheit" (default: "celsius")

**Returns:**
```json
{
  "location": "string",
  "current": {
    "temperature": "number",
    "humidity": "number",
    "rainfall": "number",
    "wind_speed": "number",
    "conditions": "string"
  },
  "forecast": [
    {
      "date": "string",
      "min_temp": "number",
      "max_temp": "number",
      "rainfall_mm": "number",
      "humidity": "number",
      "conditions": "string"
    }
  ]
}
```

---

### 2. get_historical_weather
Retrieve past weather patterns for seasonal planning.

**Parameters:**
- `location` (string, required): Location name or coordinates
- `start_date` (string, required): Start date (YYYY-MM-DD)
- `end_date` (string, required): End date (YYYY-MM-DD)
- `metrics` (array, optional): Specific metrics to retrieve ["temperature", "rainfall", "humidity"]

**Returns:**
```json
{
  "location": "string",
  "period": "string",
  "data": [
    {
      "date": "string",
      "temperature": "number",
      "rainfall": "number",
      "humidity": "number"
    }
  ],
  "summary": {
    "avg_temperature": "number",
    "total_rainfall": "number",
    "avg_humidity": "number"
  }
}
```

---

### 3. get_climate_alerts
Fetch extreme weather warnings and alerts.

**Parameters:**
- `location` (string, required): Location name or coordinates
- `alert_types` (array, optional): Filter by alert types ["storm", "drought", "flood", "frost", "heatwave"]

**Returns:**
```json
{
  "location": "string",
  "active_alerts": [
    {
      "type": "string",
      "severity": "string",
      "title": "string",
      "description": "string",
      "start_time": "string",
      "end_time": "string",
      "recommendations": ["string"]
    }
  ]
}
```

---

## Soil & Land Tools

### 4. get_soil_conditions
Retrieve soil data for a specific location.

**Parameters:**
- `location` (string, required): Location name or coordinates
- `depth` (string, optional): Soil depth - "topsoil" or "subsoil" (default: "topsoil")

**Returns:**
```json
{
  "location": "string",
  "soil_type": "string",
  "ph_level": "number",
  "moisture_content": "number",
  "organic_matter": "number",
  "nutrients": {
    "nitrogen": "string",
    "phosphorus": "string",
    "potassium": "string"
  },
  "texture": "string",
  "drainage": "string"
}
```

---

### 5. analyze_soil_health
Get comprehensive soil health assessment and recommendations.

**Parameters:**
- `location` (string, required): Location name or coordinates
- `crop_type` (string, optional): Intended crop for targeted recommendations
- `test_results` (object, optional): Recent soil test results if available

**Returns:**
```json
{
  "location": "string",
  "health_score": "number",
  "assessment": {
    "fertility": "string",
    "ph_status": "string",
    "nutrient_status": "string",
    "concerns": ["string"]
  },
  "recommendations": [
    {
      "priority": "string",
      "action": "string",
      "details": "string",
      "timeline": "string"
    }
  ]
}
```

---

### 6. get_irrigation_requirements
Calculate water needs based on soil and crop type.

**Parameters:**
- `location` (string, required): Location name or coordinates
- `crop_type` (string, required): Type of crop
- `soil_type` (string, optional): Soil type if known
- `growth_stage` (string, optional): Current growth stage of crop

**Returns:**
```json
{
  "crop": "string",
  "location": "string",
  "daily_water_requirement": "number",
  "irrigation_frequency": "string",
  "method_recommendations": ["string"],
  "seasonal_adjustment": {
    "current_needs": "string",
    "upcoming_changes": "string"
  },
  "water_conservation_tips": ["string"]
}
```

---

## Crop Management Tools

### 7. get_crop_calendar
Get optimal planting and harvesting schedules for specific crops.

**Parameters:**
- `crop_type` (string, required): Type of crop
- `location` (string, required): Location name or coordinates
- `variety` (string, optional): Specific crop variety
- `year` (integer, optional): Target year (default: current year)

**Returns:**
```json
{
  "crop": "string",
  "variety": "string",
  "location": "string",
  "calendar": {
    "land_preparation": {
      "start": "string",
      "end": "string",
      "activities": ["string"]
    },
    "planting": {
      "optimal_start": "string",
      "optimal_end": "string",
      "considerations": ["string"]
    },
    "growing_stages": [
      {
        "stage": "string",
        "duration_days": "number",
        "key_activities": ["string"]
      }
    ],
    "harvesting": {
      "expected_start": "string",
      "expected_end": "string",
      "indicators": ["string"]
    }
  }
}
```

---

### 8. get_pest_disease_info
Identify and get treatment advice for pests and diseases.

**Parameters:**
- `crop_type` (string, required): Type of crop affected
- `symptoms` (array, optional): Observed symptoms
- `pest_name` (string, optional): Name of pest/disease if known
- `location` (string, required): Location for region-specific advice

**Returns:**
```json
{
  "crop": "string",
  "common_threats": [
    {
      "name": "string",
      "type": "string",
      "symptoms": ["string"],
      "severity": "string",
      "prevention": ["string"],
      "treatment": {
        "organic_methods": ["string"],
        "chemical_methods": ["string"],
        "timing": "string"
      },
      "season_risk": "string"
    }
  ],
  "current_outbreaks": ["string"]
}
```

---

### 9. get_crop_recommendations
Suggest suitable crops based on conditions.

**Parameters:**
- `location` (string, required): Location name or coordinates
- `season` (string, optional): Target season
- `soil_type` (string, optional): Soil type if known
- `farm_size` (string, optional): Size of farm - "small", "medium", "large"
- `resources` (object, optional): Available resources (irrigation, equipment, capital)

**Returns:**
```json
{
  "location": "string",
  "season": "string",
  "recommended_crops": [
    {
      "crop_name": "string",
      "suitability_score": "number",
      "reasons": ["string"],
      "expected_yield": "string",
      "market_potential": "string",
      "input_requirements": {
        "water": "string",
        "fertilizer": "string",
        "labor": "string"
      },
      "profitability_estimate": "string"
    }
  ]
}
```

---

### 10. get_fertilizer_advice
Recommend fertilizer types and application schedules.

**Parameters:**
- `crop_type` (string, required): Type of crop
- `location` (string, required): Location name or coordinates
- `soil_conditions` (object, optional): Current soil test results
- `growth_stage` (string, optional): Current growth stage

**Returns:**
```json
{
  "crop": "string",
  "location": "string",
  "recommendations": [
    {
      "fertilizer_type": "string",
      "composition": "string",
      "application_rate": "string",
      "application_method": "string",
      "timing": "string",
      "growth_stage": "string",
      "cost_estimate": "string"
    }
  ],
  "organic_alternatives": ["string"],
  "precautions": ["string"]
}
```

---

## Market & Pricing Tools

### 11. get_crop_prices
Fetch current market prices for crops in specific regions.

**Parameters:**
- `crop_type` (string, required): Type of crop
- `location` (string, required): Location or market name
- `unit` (string, optional): Price unit - "kg", "ton", "bag" (default: "kg")
- `grade` (string, optional): Crop grade/quality

**Returns:**
```json
{
  "crop": "string",
  "location": "string",
  "date": "string",
  "prices": {
    "current_price": "number",
    "unit": "string",
    "currency": "string",
    "grade": "string"
  },
  "price_range": {
    "minimum": "number",
    "maximum": "number",
    "average": "number"
  },
  "market_status": "string",
  "demand_level": "string"
}
```

---

### 12. get_price_trends
Analyze historical price patterns for market timing.

**Parameters:**
- `crop_type` (string, required): Type of crop
- `location` (string, required): Location or market name
- `period` (string, optional): Time period - "month", "quarter", "year" (default: "year")
- `years_back` (integer, optional): Historical years to analyze (default: 3)

**Returns:**
```json
{
  "crop": "string",
  "location": "string",
  "analysis_period": "string",
  "trends": [
    {
      "period": "string",
      "avg_price": "number",
      "high_price": "number",
      "low_price": "number"
    }
  ],
  "seasonal_patterns": {
    "peak_months": ["string"],
    "low_months": ["string"]
  },
  "forecast": {
    "next_month": "string",
    "next_quarter": "string",
    "confidence": "string"
  },
  "recommendations": ["string"]
}
```

---

### 13. get_nearby_markets
Find local markets and their operating schedules.

**Parameters:**
- `location` (string, required): Current location
- `radius_km` (number, optional): Search radius in kilometers (default: 50)
- `market_type` (array, optional): Types - ["wholesale", "retail", "farmers_market"]
- `crop_type` (string, optional): Filter markets dealing in specific crops

**Returns:**
```json
{
  "location": "string",
  "markets": [
    {
      "name": "string",
      "type": "string",
      "distance_km": "number",
      "address": "string",
      "operating_days": ["string"],
      "operating_hours": "string",
      "crops_traded": ["string"],
      "facilities": ["string"],
      "contact": "string"
    }
  ]
}
```

---

## Knowledge & Best Practices Tools

### 14. get_farming_practices
Access farming techniques and best practices for specific crops.

**Parameters:**
- `crop_type` (string, required): Type of crop
- `practice_category` (string, optional): Category - "planting", "cultivation", "harvesting", "post-harvest"
- `farming_method` (string, optional): Method - "organic", "conventional", "sustainable"
- `location` (string, optional): Location for region-specific practices

**Returns:**
```json
{
  "crop": "string",
  "category": "string",
  "practices": [
    {
      "title": "string",
      "description": "string",
      "steps": ["string"],
      "benefits": ["string"],
      "resources_needed": ["string"],
      "difficulty_level": "string",
      "cost_estimate": "string"
    }
  ],
  "regional_adaptations": ["string"],
  "common_mistakes": ["string"]
}
```

---

### 15. get_seasonal_advice
Get season-specific farming guidance.

**Parameters:**
- `location` (string, required): Location name or coordinates
- `season` (string, optional): Season name or auto-detect current season
- `crop_types` (array, optional): Specific crops of interest

**Returns:**
```json
{
  "location": "string",
  "season": "string",
  "period": "string",
  "key_activities": [
    {
      "activity": "string",
      "priority": "string",
      "timing": "string",
      "details": "string"
    }
  ],
  "crop_specific_advice": [
    {
      "crop": "string",
      "recommendations": ["string"]
    }
  ],
  "weather_considerations": ["string"],
  "preparation_tips": ["string"]
}
```

---

### 16. search_agricultural_knowledge
Search database of farming guides and resources.

**Parameters:**
- `query` (string, required): Search query
- `category` (array, optional): Categories to search ["crops", "livestock", "equipment", "techniques", "diseases"]
- `language` (string, optional): Preferred language (default: "en")
- `limit` (integer, optional): Number of results (default: 10)

**Returns:**
```json
{
  "query": "string",
  "results": [
    {
      "title": "string",
      "category": "string",
      "summary": "string",
      "content": "string",
      "source": "string",
      "relevance_score": "number",
      "language": "string",
      "related_topics": ["string"]
    }
  ],
  "total_results": "number"
}
```

---

## Location & Farm Tools

### 17. get_farm_profile
Retrieve or update farmer's profile information.

**Parameters:**
- `farmer_id` (string, required): Unique farmer identifier
- `action` (string, required): Action - "get" or "update"
- `profile_data` (object, optional): Profile data for updates

**Returns:**
```json
{
  "farmer_id": "string",
  "name": "string",
  "location": "string",
  "contact": "string",
  "farm_details": {
    "total_area": "number",
    "area_unit": "string",
    "soil_type": "string",
    "irrigation_available": "boolean",
    "equipment": ["string"]
  },
  "crops": [
    {
      "crop_type": "string",
      "area_planted": "number",
      "planting_date": "string"
    }
  ],
  "preferences": {
    "language": "string",
    "notification_method": "string",
    "farming_method": "string"
  }
}
```

---

### 18. calculate_yield_estimate
Estimate potential crop yields based on current conditions.

**Parameters:**
- `crop_type` (string, required): Type of crop
- `location` (string, required): Location name or coordinates
- `farm_area` (number, required): Area planted
- `area_unit` (string, required): Unit - "hectare", "acre"
- `planting_date` (string, required): Date crop was planted (YYYY-MM-DD)
- `inputs` (object, optional): Applied inputs (fertilizer, irrigation, etc.)

**Returns:**
```json
{
  "crop": "string",
  "location": "string",
  "planted_area": "number",
  "estimates": {
    "expected_yield": "number",
    "yield_unit": "string",
    "confidence_level": "string",
    "yield_per_unit_area": "number"
  },
  "factors": {
    "weather_impact": "string",
    "soil_quality_impact": "string",
    "management_impact": "string"
  },
  "harvest_projection": {
    "expected_date": "string",
    "total_production": "number",
    "market_value_estimate": "number"
  },
  "recommendations_to_improve": ["string"]
}
```

---

## Translation & Communication Tools

### 19. translate_query
Translate questions from local languages to English.

**Parameters:**
- `text` (string, required): Text to translate
- `source_language` (string, required): Source language code (e.g., "sw", "ha", "yo", "am")
- `preserve_context` (boolean, optional): Preserve agricultural context (default: true)

**Returns:**
```json
{
  "original_text": "string",
  "translated_text": "string",
  "source_language": "string",
  "target_language": "string",
  "confidence": "number",
  "detected_intent": "string",
  "key_terms": ["string"]
}
```

---

### 20. translate_response
Translate AI advice back to local languages.

**Parameters:**
- `text` (string, required): Text to translate
- `target_language` (string, required): Target language code
- `simplify` (boolean, optional): Use simpler language (default: true)
- `include_audio` (boolean, optional): Generate audio version (default: false)

**Returns:**
```json
{
  "original_text": "string",
  "translated_text": "string",
  "target_language": "string",
  "audio_url": "string",
  "readability_level": "string"
}
```

---

## Notes

- All date parameters follow ISO 8601 format (YYYY-MM-DD)
- Location can be specified as city name, coordinates (lat,lon), or postal code
- All numeric measurements can be returned in metric or imperial units based on regional settings
- Error responses follow standard format:
```json
{
  "error": "string",
  "code": "string",
  "message": "string",
  "suggestions": ["string"]
}
```