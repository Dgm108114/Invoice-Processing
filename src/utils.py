import re
import logging

# setting logging
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Patterns for matching
idv_pattern = [
    r"IDV\s*₹\s*([\d,]+\.\d{2})",
    r"Total IDV\s*₹\s*([\d,]+\.\d{2})",
    r"Total IDV\s*\(₹\)\s*([\d,]+\.\d{2})",
    r"[\d,.]+"
    ]

total_premium_patterns = [
    r"Total Premium\s*₹\s*([\d,]+\.\d{2})",
    r"TOTAL PREMIUM\s*([\d,]+\.\d{2})",
    r"TOTAL PREMIUM PAYABLE\s*\(₹\)\s*([\d,]+\.\d{2})",
    r"Total PREMIUM Payable in\s*₹\s*([\d,]+\.\d{2})",
    r"TOTAL POLICY PREMIUM\s*([\d,]+\.\d{2})",
    r"Total Policy Premium\s*([\d,]+\.\d{2})",
    r"[\d,.]+"
]

own_damage_patterns = [
    r"TOTAL OWN DAMAGE PREMIUM\s*([\d,]+\.\d{2})",
    r"Total Own Damage Premium\(A\)\s*([\d,]+\.\d{2})",
    r"[\d,.]+"
]

ncb_patterns = [
    r"Deduct\s*(\d+)% for NCB",
    r"Deduct\s*(\d+)\s*% for NCB",
    r"No Claim Bonus\s*(\d+)%",
    r"Less:No claim bonus\s*\((\d+)%\)",
    r"Current Year NCB\s*\(%\)\s*(\d+)%"
]

# # Function to extract IDV
# def extract_idv(text):
#     try:
#         logger.info("Applying regex pattern for IDV")
#         for pattern in idv_pattern:
#             match = re.search(pattern, text)
#             if match:
#                 return match.group(1)
#             else:
#                 logger.info("No match found for IDV")
#                 return None
#     except Exception as e:
#         logger.error(f"Error applying regex on IDV: {e}")
#         logger.debug(traceback.format_exc())
#         return None

def extract_idv(text):
    try:
        logger.info("Applying regex pattern for IDV")
        idv_values = re.findall(r"[\d,.]+",text)
        idv_value = ''
        for i in idv_values:
            if len(i)>=4:
                idv_value = i
            else:
                pass
        print("idv_value : ", idv_value)
        return idv_value
    except Exception as e:
        logger.error(f"Error applying regex on IDV: {e}")
        logger.debug(traceback.format_exc())
        return None

# def extract_total_premium_payable(text):
#     try:
#         logger.info("Applying regex pattern for Total premium")
#         for pattern in total_premium_patterns:
#             match = re.search(pattern, text)
#             if match:
#                 return match.group(1)
#             else:
#                 logger.info("No match found for Total premium")
#                 return None
#     except Exception as e:
#         logger.info(f"Error applying regex on Total premium: {e}")
#         logger.debug(traceback.format_exc())
#         return None


def extract_total_premium_payable(text):
    try:
        logger.info("Applying regex pattern for Total premium")
        total_premiums = re.findall(r"[\d,.]+",text)
        total_premium = ''
        for i in total_premiums:
            if len(i)>=4:
                total_premium = i
            else:
                pass
        print("total_premium : ", total_premium)
        return total_premium
    except Exception as e:
        logger.info(f"Error applying regex on Total premium: {e}")
        logger.debug(traceback.format_exc())
        return None

# def extract_own_damage(text):
#     try:
#         logger.info("applying regex pattern for Own damage")
#         for pattern in own_damage_patterns:
#             match = re.search(pattern, text)
#             if match:
#                 return match.group(1)
#             else:
#                 logger.info("No match found for for Own damage")
#     except Exception as e:
#         logger.error(f"Error applying regex on for Own damage: {e}")
#         logger.debug(traceback.format_exc())
#         return None


def extract_own_damage(text):
    try:
        logger.info("applying regex pattern for Own damage")
        own_damages = re.findall(r"[\d,.]+",text)
        own_damage = ''
        for i in own_damages:
            if len(i)>=4:
                own_damage = i
            else:
                pass
        print("own_damage : ", own_damage)
        return own_damage
    except Exception as e:
        logger.error(f"Error applying regex on for Own damage: {e}")
        logger.debug(traceback.format_exc())
        return None

def extract_policy_name(text):
    try:
        logger.info("Applying regex pattern for Policy name")
        if not (extract_idv(text) or extract_total_premium_payable(text) or extract_own_damage(text) or extract_ncb(text)):
            match = re.search(r"^\s*(.*)\s*$", text)
            if match:
                return match.group(1).strip()
    except Exception as e:
        logger.error(f"Error applying regex on Policy name: {e}")
        logger.debug(traceback.format_exc())
        return None


def extract_ncb(text):
    try:
        logger.info("Applying regex pattern for NCB")
        for pattern in ncb_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
            else:
                ncb_values = re.findall(r"[\d,.]+", text)
                for ncb_value in ncb_values:
                    if len(ncb_value)==2:
                        return ncb_value
                else:
                    return None
    except Exception as e:
        logger.error(f"Error applying regex on NCB: {e}")
        logger.debug(traceback.format_exc())
        return None


def apply_regex(text, class_name):
    try:
        if class_name == "IDV":
            idv_value = extract_idv(text)
            return ("SumInsured".upper(), idv_value)

        elif class_name == "NCB":
            ncb_value = extract_ncb(text)
            return ("NCB", ncb_value)

        elif class_name == "company-name":
            company_name = text
            return ("InsurerSAIBA".upper(), company_name)

        elif class_name == "product-name":
            product_name = text
            return ("ProductName".upper(), product_name)

        elif class_name == "total-premium":
            total_premium = extract_total_premium_payable(text)
            return ("GrossPrem".upper(), total_premium)

        elif class_name == "own-damage":
            own_damage = extract_own_damage(text)
            return ("ODNetPremium".upper(), own_damage)

        else:
            logger.info("No class detected")
            return None

    except Exception as e:
        logger.error(f"Error applying apply_regex function: {e}")
        logger.debug(traceback.format_exc())
        return None


def clean_text(text):
    try:
        logger.info("Applying preprocessing on text")
        # 1. Remove Email
        text = re.sub(r'<////>', 'EPICENTER', text)
        # 1. Remove Email
        cleaned_text = re.sub(r'\b[\w\-\.]+@[\w\.-]+\b', ' ', text)
        # 2. Remove From Bracket
        cleaned_text = re.sub(r"[\(\[].*?[\)\]]", "", cleaned_text)
        # 3. Add space Before and after (')
        cleaned_text = re.sub(r"'", " ", cleaned_text)
        # 4. Remove double repeating special characters
        cleaned_text = re.sub(r'([^\w\s])\1', r'\1', cleaned_text)
        # 5. Add space after ' ' '&' using regex
        cleaned_text = re.sub(r"(,|&)", r"\1 ", cleaned_text)
        # 6. Add space Before and after '/'
        cleaned_text = re.sub(r"/", " / ", cleaned_text)
        # Replace "None" and '\n' with a space
        cleaned_text = re.sub(r'None|\\n', ' ', cleaned_text)
        # Replace '.' if it's followed by a word character
        cleaned_text = re.sub(r'\.(?=\w)', ' ', cleaned_text)
        cleaned_text = re.sub(r"[\s]+", " ", cleaned_text)
        cleaned_text = re.sub(r'EPICENTER', '<////>', cleaned_text)

        return cleaned_text
    except Exception as e:
        logger.error(f"Error in applying preprocessing on text: {e}")
        logger.debug(traceback.format_exc())
        return None
