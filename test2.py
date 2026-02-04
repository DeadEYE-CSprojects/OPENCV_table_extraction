import os
import shutil
import pandas as pd
import re
import cv2
import numpy as np
import pytesseract
import subprocess
import json
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from openai import OpenAI
import openpyxl
from docx2pdf import convert as docx_to_pdf_convert
from PIL import Image
import time
import base64
import argparse
