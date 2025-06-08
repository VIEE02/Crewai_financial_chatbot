#!/usr/bin/env python
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from typing import Optional
import os
from datetime import datetime
from .crew import Financial
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
# Load environment variables
app = FastAPI()

class ChatRequest(BaseModel):
    user_query: str
def format_response(result_dict: dict) -> str:
    """Format JSON result into a readable response using the data"""
    try:
        if "answer" in result_dict:
            # Already have a formatted answer
            return result_dict["answer"]
            
        if "metadata" in result_dict:
            data = result_dict["metadata"]
            date = data.get("date", "")
            
            # Get data points
            points = data.get("data_points", {})
            
            # Format response based on available data
            response_parts = []
            
            # Add closing prices if available
            for ticker, info in points.items():
                if isinstance(info, dict) and "closing_price" in info:
                    response_parts.append(f"{ticker} có giá đóng cửa là ${info['closing_price']:.2f}")
                    if "opening_price" in info:
                        response_parts.append(f"và giá mở cửa là ${info['opening_price']:.2f}")
            
            # Add highest/lowest if available
            if "highest_closing" in points:
                highest = points["highest_closing"]
                response_parts.append(
                    f"Giá cao nhất thuộc về {highest['ticker']} với ${highest['price']:.2f}"
                )
            
            if "lowest_price" in points:
                response_parts.append(f"Giá thấp nhất là ${points['lowest_price']:.2f}")
            
            # Combine all parts
            if response_parts:
                response = f"Vào ngày {date}, " + ". ".join(response_parts) + "."
                return response
            
        return "Không tìm thấy dữ liệu phù hợp."
        
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        return "Lỗi khi định dạng câu trả lời."
def get_crew_response(user_query: str) -> dict:
    """Get response from CrewAI agents"""
    try:
        # Extract company symbol from query
        import re
        symbols = re.findall(r'\b[A-Z]{2,5}\b', user_query.upper())
        company_1 = symbols[0] if symbols else None
        company_2 = symbols[1] if len(symbols) > 1 else None
        
        # Extract price field from query (open, close, high, low)
        price_field = None
        if any(word in user_query.lower() for word in ['open', 'opened', 'opening']):
            price_field = 'open'
        elif any(word in user_query.lower() for word in ['close', 'closed', 'closing']):
            price_field = 'close'
        elif any(word in user_query.lower() for word in ['high', 'highest']):
            price_field = 'high'
        elif any(word in user_query.lower() for word in ['low', 'lowest']):
            price_field = 'low'

        inputs = {
            'api_endpoint': os.getenv('api_url', 'http://localhost:8000'),
            'user_message': user_query,
            'company_1': company_1 or 'unknown',  # Provide default value
            'company_2': company_2 or 'unknown',  # Provide default value
            'stock_prices_dir': os.getenv('stock_data_folder_path'),
            'company_csv_path': os.getenv('companies_csv_path'),
            'timestamp': datetime.now().isoformat(),
            'system_prompt': os.getenv('system_prompt'),
            'date': datetime.now().strftime("%Y-%m-%d"),
            'price_field': price_field or 'close',  # Default to 'close' if not specified
            # Add all other required template variables with default values
            'llm_query_analysis_task': 'Analyze the query',
            'historical_data_loading_task': 'Load historical data',
            'historical_query_processing_task': 'Process historical data'
        }

        logger.info(f"Extracted companies: {company_1}, {company_2}")
        logger.info(f"Extracted price field: {price_field}")
        logger.info(f"Processing with inputs: {inputs}")

        # Initialize and run the CrewAI
        crew = Financial().crew()
        result = crew.kickoff(inputs=inputs)
        
        # Log the raw result
        logger.info(f"CrewAI raw result: {result}")

        if hasattr(result, 'result'):
            crew_result = result.result
            if isinstance(crew_result, dict):
                # Format the response
                formatted_response = format_response(crew_result)
                return {"response": formatted_response}
            elif isinstance(crew_result, str):
                return {"response": crew_result}
            else:
                logger.info(f"Crew result content: {crew_result}")
                return {"response": str(crew_result)}
        else:
            logger.error(f"Unexpected result structure: {result}")
            formatted_response = format_response(result)
            return {"response": formatted_response}
        
    except Exception as e:
        logger.error(f"Error in get_crew_response: {str(e)}", exc_info=True)
        return {"response": f"Lỗi xử lý: {str(e)}"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_query = data.get("user_query")
        
        if not user_query:
            return JSONResponse(
                content={"response": "Vui lòng nhập câu hỏi"},
                status_code=400
            )

        logger.info(f"Received query: {user_query}")
        result = get_crew_response(user_query)
        logger.info(f"Final response: {result}")
        
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"response": f"Lỗi hệ thống: {str(e)}"},
            status_code=500
        )

def run():
    """
    Run the crew.
    """
    inputs = {
        'api_endpoint': 'http://localhost:8000',
        'user_message': 'sample_value',
        'company_1': 'sample_value',
        'company_2': 'sample_value',
        'date': 'sample_value',
        'price_field': 'sample_value',
        'stock_prices_dir': r'C:\Users\Admin\Desktop\CrewAI\financial\stock_prices',
        'company_csv_path': r'C:\Users\Admin\Desktop\CrewAI\financial\djia_companies_20250524.csv',
        'timestamp': 'sample_value',
        'llm_query_analysis_task': 'sample_value',
        'historical_data_loading_task': 'sample_value',
        'historical_query_processing_task': 'sample_value'
    }
    Financial().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'api_endpoint': 'http://localhost:8000',
        'user_message': 'sample_value',
        'company_1': 'sample_value',
        'company_2': 'sample_value',
        'date': 'sample_value',
        'price_field': 'sample_value',
        'stock_prices_dir': r'C:\Users\Admin\Desktop\CrewAI\financial\stock_prices',
        'company_csv_path': r'C:\Users\Admin\Desktop\CrewAI\financial\djia_companies_20250524.csv',
        'timestamp': 'sample_value',
        'llm_query_analysis_task': 'sample_value',
        'historical_data_loading_task': 'sample_value',
        'historical_query_processing_task': 'sample_value'
    }
    try:
        Financial().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Financial().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'api_endpoint': 'http://localhost:8000',
        'user_message': 'sample_value',
        'company_1': 'sample_value',
        'company_2': 'sample_value',
        'date': 'sample_value',
        'price_field': 'sample_value',
        'stock_prices_dir': r'C:\Users\Admin\Desktop\CrewAI\financial\stock_prices',
        'company_csv_path': r'C:\Users\Admin\Desktop\CrewAI\financial\djia_companies_20250524.csv',
        'timestamp': 'sample_value',
        'llm_query_analysis_task': 'sample_value',
        'historical_data_loading_task': 'sample_value',
        'historical_query_processing_task': 'sample_value'
    }
    try:
        Financial().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
