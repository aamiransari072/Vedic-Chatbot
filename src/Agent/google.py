import google.generativeai as genai
from typing import Dict, Any, Optional


class Gemini:
    def __init__(self, api_key='api_key', id='gemini-1.5-flash-latest', temprature=0.2, **kwargs):
        self.api_key = api_key
        self.id = id
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            self.id,
            generation_config=genai.GenerationConfig(
                temperature=temprature,
                **kwargs
            )
        )
    
    def generate(self, prompt):
        """
        Generate content using Gemini model
        
        Args:
            prompt: The prompt string or object to send to the model
            
        Returns:
            GeminiResponse: A wrapper object with the model's response
        """
        response = self.model.generate_content([prompt])
        return GeminiResponse(response)


class GeminiResponse:
    """
    Wrapper class for Gemini responses to provide a consistent interface
    """
    
    def __init__(self, response):
        """
        Initialize with a raw Gemini response
        
        Args:
            response: The raw response from the Gemini model
        """
        self.raw_response = response
        
    @property
    def text(self) -> str:
        """
        Get the text content of the response
        
        Returns:
            The text content as a string
        """
        try:
            # Handle different response formats
            if hasattr(self.raw_response, "text"):
                return self.raw_response.text
            elif hasattr(self.raw_response, "parts"):
                return "".join(part.text for part in self.raw_response.parts)
            elif hasattr(self.raw_response, "candidates"):
                candidates = self.raw_response.candidates
                if candidates and len(candidates) > 0:
                    parts = candidates[0].content.parts
                    return "".join(part.text for part in parts)
            
            # Fallback: convert to string
            return str(self.raw_response)
            
        except Exception as e:
            # Return error message if parsing fails
            return f"Error extracting text from response: {str(e)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response to a dictionary
        
        Returns:
            Dictionary representation of the response
        """
        return {
            "text": self.text,
            # Add more properties as needed
        }




    