from langgraph.store.base import Op
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbconvert.preprocessors import ExecutePreprocessor
import os
from typing import Literal, Optional, Dict, Any
from langchain_core.tools import tool

from pydantic import BaseModel, Field
import copy
import base64
import requests
from openai import AzureOpenAI
import logging
import subprocess
logger = logging.getLogger(__name__)


class VisionAnalyzerConfig(BaseModel):
    provider: Literal["azure_openai", "nvidia_build"] = Field(description="The provider of the vision analyzer to use.")
    model_name: str = Field(description="The model name to use.")
    api_key: str = Field(description="The API key to use.", default=None)
    api_base: str = Field(description="The API base to use.", default=None)
    api_version: str = Field(description="The API version to use.", default=None)


class NotebookManager:
    """Manages Jupyter notebook creation, modification, and execution."""
    
    def __init__(self, notebook_path: str, vision_config: VisionAnalyzerConfig):
        self.notebook_path = notebook_path
        self.notebook = None

        # Initialize vision analyzer
        if vision_config.provider == "azure_openai":
            self.vision_analyzer = OpenAIVisionAnalyzer(vision_config)
        elif vision_config.provider == "nvidia_build":
            self.vision_analyzer = NVBuildVisionAnalyzer(vision_config)
        else:
            raise NotImplementedError(f"Vision analyzer type: {vision_config.provider} not supported")

        # either load existing notebook or create new one
        if os.path.exists(self.notebook_path):
            with open(self.notebook_path, 'r') as f:
                self.notebook = nbformat.read(f, as_version=4)
        else:
            self.notebook = new_notebook()
            self._save_notebook()
    
    def _save_notebook(self):
        """Save notebook to disk."""
        os.makedirs(os.path.dirname(self.notebook_path), exist_ok=True)
        with open(self.notebook_path, 'w') as f:
            nbformat.write(self.notebook, f)
    
    def append_cell(self, content: str, cell_type: Literal["code", "markdown"] = "code") -> Dict[str, Any]:
        """Append a new cell to the notebook and execute it."""
        if cell_type == "code":
            cell = new_code_cell(content)
        else:
            cell = new_markdown_cell(content)
        
        self.notebook.cells.append(cell)
        self._save_notebook()
        
        # Execute and get output of the new cell
        cell_index = len(self.notebook.cells) - 1
        return self._execute_and_get_output(cell_index)
    
    def modify_last_cell(self, content: str, cell_type: Literal["code", "markdown"] = "code") -> Dict[str, Any]:
        """Modify the last cell in the notebook and execute it."""
        if len(self.notebook.cells) == 0:
            return self.append_cell(content, cell_type)
        
        last_cell_index = len(self.notebook.cells) - 1
        return self.modify_cell(last_cell_index, content, cell_type)
        
    
    def modify_cell(self, cell_index: int, content: str, cell_type: Literal["code", "markdown"] = "code") -> Dict[str, Any]:
        """Modify a specific cell by index and execute it."""
        if cell_index < 0 or cell_index >= len(self.notebook.cells):
            return {
                "success": False,
                "error": f"Cell index {cell_index} out of range. Notebook has {len(self.notebook.cells)} cells.",
                "output": "",
                "cell_index": cell_index
            }
        
        if cell_type == "code":
            self.notebook.cells[cell_index] = new_code_cell(content)
        else:
            self.notebook.cells[cell_index] = new_markdown_cell(content)
        
        self._save_notebook()
        return self._execute_and_get_output(cell_index)
    
    def delete_cell(self, cell_index: int) -> Dict[str, Any]:
        """Delete a cell by index and execute the entire notebook."""
        if cell_index < 0 or cell_index >= len(self.notebook.cells):
            return {
                "success": False,
                "error": f"Cell index {cell_index} out of range. Notebook has {len(self.notebook.cells)} cells.",
            }
        self.notebook.cells.pop(cell_index)
        self._save_notebook()
        return {"success": True, "output": "", "error": None}
    
    def _execute_and_get_output(self, target_cell_index: int) -> Dict[str, Any]:
        """
        Execute the entire notebook and return only the output of the target cell.
        
        Args:
            target_cell_index: Index of the cell whose output should be captured
            
        Returns:
            Dictionary with success status, output, and error information.
            If an error occurred, error_cell_index will indicate which cell failed.
        """
        # try:
        
        # Create a copy of the notebook for execution
        exec_notebook = copy.deepcopy(self.notebook)
        
        # Execute the notebook with allow_errors=True to continue through errors
        # This allows us to detect which specific cell failed
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3', allow_errors=True)
        ep.preprocess(exec_notebook)
        
        # Update the original notebook with execution results
        self.notebook = exec_notebook
        self._save_notebook()
        
        # Check all cells for errors to find which one failed
        error_cell_index = None
        error_info = None
        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
                for output in cell.outputs:
                    if output.output_type == 'error':
                        error_cell_index = i
                        error_info = {
                            'ename': output.ename,
                            'evalue': output.evalue,
                            'traceback': '\n'.join(output.traceback)
                        }
                        # Found the first error, stop searching
                        break
            if error_cell_index is not None:
                break
        
        # Extract output from the target cell
        target_cell = self.notebook.cells[target_cell_index]
        output_text = self._extract_cell_output(target_cell)
        
        # If there was an error, report it with the correct cell index
        if error_info is not None:
            # print("error_info is:", error_info)
            
            return {
                "success": False,
                "output": output_text,
                "error": error_info['traceback'],
                "error_cell_index": error_cell_index,
                "cell_index": target_cell_index,
                "cell_type": target_cell.cell_type,
                "total_cells": len(self.notebook.cells)
            }
        
        return {
            "success": True,
            "output": output_text,
            "error": None,
            "cell_index": target_cell_index,
            "cell_type": target_cell.cell_type,
            "total_cells": len(self.notebook.cells)
        }
    
    def _extract_cell_output(self, cell) -> str:
        """Extract text output from a cell."""
        if cell.cell_type != "code":
            return f"[Markdown cell - no output]"
        
        if not hasattr(cell, 'outputs') or len(cell.outputs) == 0:
            return "[No output]"
        
        output_parts = []
        for output in cell.outputs:
            if output.output_type == 'stream':
                output_parts.append(output.text)
            elif output.output_type == 'execute_result':
                if 'text/plain' in output.data:
                    output_parts.append(output.data['text/plain'])
                # Check for images in execute_result
                if 'image/png' in output.data:
                    image_analysis = self._analyze_image_output(output.data['image/png'])
                    output_parts.append(f"Plot Description: {image_analysis}")
            elif output.output_type == 'display_data':
                if 'text/plain' in output.data:
                    output_parts.append(output.data['text/plain'])
                # Analyze PNG images using vision model
                if 'image/png' in output.data:
                    image_analysis = self._analyze_image_output(output.data['image/png'])
                    output_parts.append(f"Plot Description: {image_analysis}")
            elif output.output_type == 'error':
                output_parts.append(f"Error: {output.ename}: {output.evalue}")
                output_parts.append('\n'.join(output.traceback))
        
        return '\n'.join(output_parts) if output_parts else "[No output]"
    
    def _analyze_image_output(self, image_base64: str) -> str:
        """
        Analyze an image output using the vision model.
        
        Args:
            image_base64: Base64-encoded PNG image from notebook output
            
        Returns:
            Vision analysis result or fallback message
        """
        if self.vision_analyzer is None:
            return "[Image output generated - Vision analysis disabled]"
        
        try:
            # Decode the base64 image
            image_data = base64.b64decode(image_base64)
            
            # Analyze with vision model
            logger.info("  📊 Analyzing plot with vision model...")
            analysis = self.vision_analyzer.analyze_plot(image_data)
            return analysis
            
        except Exception as e:
            logger.warning(f"  Warning: Failed to analyze image: {str(e)}")
            return "[Image output generated - Vision analysis failed]"
    
    def get_notebook_summary(self) -> str:
        """Get a summary of the notebook structure."""
        summary = f"Notebook: {self.notebook_path}\n"
        summary += f"Total cells: {len(self.notebook.cells)}\n\n"
        
        for i, cell in enumerate(self.notebook.cells):
            cell_content = cell.source
            summary += f"Cell {i} ({cell.cell_type}): {cell_content}\n"
        
        return summary


class NVBuildVisionAnalyzer:
    """Analyzes plots and visualizations using a vision language model."""
    
    def __init__(self, config: VisionAnalyzerConfig):
        """
        Initialize the vision analyzer.
        
        Args:
            config: Vision analyzer configuration.
        """
        self.api_key = config.api_key
        self.invoke_url = config.api_base
        self.model = config.model_name
    
    def analyze_plot(self, image_data: bytes) -> str:
        """
        Analyze a plot image and provide feedback.
        
        Args:
            image_data: Raw bytes of the PNG image
            
        Returns:
            Analysis and feedback from the vision model
        """
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode()
            
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": f'Describe this plot from a Jupyter notebook in 2-3 sentences. '
                                   f'Focus on what data is shown and any notable patterns. '
                                   f'If the plot quality could be improved (labels, colors, clarity, etc.), '
                                   f'provide 1-2 specific suggestions. Be encouraging but direct. '
                                   f'<img src="data:image/png;base64,{image_b64}" />'
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
            
            # Make the request
            response = requests.post(self.invoke_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            # Extract the response
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                analysis = result['choices'][0]['message']['content']
                return f"[Image output generated]\nVision Analysis: {analysis}"
            else:
                return "[Image output generated - Vision analysis: No response from model]"
                
        except requests.exceptions.Timeout:
            return "[Image output generated - Vision analysis timed out]"
        except requests.exceptions.RequestException as e:
            logger.warning(f"Vision analysis failed: {str(e)}") 
            return f"[Image output generated - Vision analysis error: {str(e)[:100]}]"
        except Exception as e:
            logger.warning(f"Unexpected error in vision analysis: {str(e)}")
            return "[Image output generated - Vision analysis failed]"


class OpenAIVisionAnalyzer:
    """Analyzes plots and visualizations using a vision language model."""
    
    def __init__(self, config: VisionAnalyzerConfig):
        """
        Initialize the vision analyzer.
        """
        self.config = config
        self.client = AzureOpenAI(
            azure_endpoint=config.api_base,
            api_version=config.api_version,
            api_key=config.api_key,
        )
    
    def analyze_plot(self, image_data: bytes) -> str:
        """
        Analyze a plot image and provide feedback.
        """
        base64_image = base64.b64encode(image_data).decode()
        messages=[
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe this plot from a Jupyter notebook in 2-3 sentences. Focus on what data is shown and any notable patterns. If the plot quality could be improved (labels, colors, clarity, etc.), provide 1-2 specific suggestions. Be encouraging but direct.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            stream=False
        )
        analysis = response.choices[0].message.content
        return f"[Image output generated]\nVision Analysis: {analysis}"