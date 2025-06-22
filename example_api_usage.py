import requests
import json
import argparse

# API endpoints
API_URL = "http://localhost:8000/analyze"
EMAIL_URL = "http://localhost:8000/send-email"

# Example meeting transcript
transcript = """
Good morning everyone. Let's start our weekly product planning meeting. Today we need to discuss the roadmap for Q3, review the customer feedback from the latest release, and decide on the feature prioritization.

John: I've analyzed the customer feedback and the top three requests are: improved search functionality, better mobile responsiveness, and integration with third-party tools.

Sarah: Thanks John. Based on our development capacity, I think we can implement the improved search and mobile responsiveness in Q3, but the third-party integrations might need to wait until Q4.

Michael: I agree with Sarah. The search functionality should be our top priority as it affects all users. I can have my team start working on it next week if we approve this plan.

Sarah: That sounds good. Let's aim to have the search improvements ready by the end of August.

John: I'll prepare the detailed requirements document by this Friday and share it with Michael's team.

CEO: Great. So we've decided to prioritize search improvements and mobile responsiveness for Q3, with third-party integrations moved to Q4. Michael's team will lead the search functionality work starting next week, and John will provide requirements by Friday. Let's meet again next week to review progress.
"""

# API request
def analyze_transcript(text, model="mistral"):
    """Send transcript to API for analysis"""
    payload = {
        "text": text,
        "model": model,
        "max_chunk_size": 1500,
        "use_cache": True
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None

# Send email with analysis results
def send_email(email_address, analysis_result):
    """Send email with analysis results"""
    if not analysis_result:
        print("No analysis results to send")
        return False
    
    payload = {
        "to": email_address,
        "subject": "Meeting Analysis Results",
        "content": {
            "summary": analysis_result["results"]["summary"],
            "decisions": analysis_result["results"]["decisions"],
            "action_items": analysis_result["results"]["action_items"],
            "metadata": {
                "model": analysis_result["model_used"],
                "processing_time": analysis_result["processing_time"],
                "fallback_used": analysis_result["fallback_used"]
            }
        }
    }
    
    try:
        response = requests.post(EMAIL_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        if result.get("success"):
            print(f"Email scheduled for delivery to {email_address}")
            return True
        else:
            print(f"Error sending email: {result.get('message', 'Unknown error')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error sending email: {e}")
        return False

# Check if API is running
def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"API Status: {health_data['status']}")
            return True
        else:
            print(f"API returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("API is not running. Start it with 'python api.py'")
        return False

# List available models
def list_models():
    """List available Ollama models"""
    try:
        response = requests.get("http://localhost:8000/models")
        if response.status_code == 200:
            data = response.json()
            if "models" in data:
                print("Available models:")
                for model in data["models"]:
                    print(f"  - {model}")
            else:
                print("No models found or error retrieving models")
        else:
            print(f"Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Meeting Transcript Analyzer Example")
    parser.add_argument("--email", type=str, help="Email address to send results to")
    parser.add_argument("--model", type=str, default="mistral", help="Model to use for analysis")
    args = parser.parse_args()
    
    print("Meeting Transcript Analyzer Example")
    print("-" * 50)
    
    # Check if API is running
    if not check_api_health():
        print("Please start the API server first with 'python api.py'")
        exit(1)
    
    # List available models
    list_models()
    
    print("\nAnalyzing transcript...")
    result = analyze_transcript(transcript, model=args.model)
    
    if result:
        print("\nAnalysis Results:")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Model used: {result['model_used']}")
        print(f"Fallback used: {result['fallback_used']}")
        
        print("\nSummary:")
        for item in result['results']['summary']:
            print(f"  - {item}")
            
        print("\nDecisions:")
        if result['results']['decisions']:
            for item in result['results']['decisions']:
                print(f"  - {item}")
        else:
            print("  None identified")
            
        print("\nAction Items:")
        if result['results']['action_items']:
            for item in result['results']['action_items']:
                print(f"  - {item}")
        else:
            print("  None identified")
            
        # Send email if requested
        if args.email:
            print(f"\nSending results to {args.email}...")
            send_email(args.email, result)
    else:
        print("Failed to analyze transcript")
        
    print("\nUsage examples:")
    print("  python example_api_usage.py                          # Just analyze the transcript")
    print("  python example_api_usage.py --email user@example.com # Analyze and email results")
    print("  python example_api_usage.py --model llama3.2         # Use a different model") 