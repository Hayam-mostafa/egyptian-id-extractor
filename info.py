from datetime import datetime
from typing import Dict
def decode_egyptian_id(id_number: str) -> Dict[str, str]:
    """
    Decode Egyptian National ID number to extract information.
    
    The Egyptian National ID is a 14-digit number with the following structure:
    - Digit 1: Century (2 = 1900s, 3 = 2000s)
    - Digits 2-3: Year of birth
    - Digits 4-5: Month of birth
    - Digits 6-7: Day of birth
    - Digits 8-9: Governorate code
    - Digits 10-12: Unique sequence number
    - Digit 13: Gender (odd = male, even = female)
    - Digit 14: Check digit
    
    Args:
        id_number: 14-digit National ID number
        
    Returns:
        Dictionary with birth_date, governorate, and gender
    """
    governorates = {
        '01': 'Cairo',
        '02': 'Alexandria',
        '03': 'Port Said',
        '04': 'Suez',
        '11': 'Damietta',
        '12': 'Dakahlia',
        '13': 'Ash Sharqia',
        '14': 'Kaliobeya',
        '15': 'Kafr El - Sheikh',
        '16': 'Gharbia',
        '17': 'Monoufia',
        '18': 'El Beheira',
        '19': 'Ismailia',
        '21': 'Giza',
        '22': 'Beni Suef',
        '23': 'Fayoum',
        '24': 'El Menia',
        '25': 'Assiut',
        '26': 'Sohag',
        '27': 'Qena',
        '28': 'Aswan',
        '29': 'Luxor',
        '31': 'Red Sea',
        '32': 'New Valley',
        '33': 'Matrouh',
        '34': 'North Sinai',
        '35': 'South Sinai',
        '88': 'Foreign'
    }
    
    # Validate ID length
    if not id_number or len(id_number) != 14:
        return {
            'Birth Date': '',
            'Governorate': '',
            'Gender': ''
        }
    
    try:
        # Extract components
        century_digit = int(id_number[0])
        year = int(id_number[1:3])
        month = int(id_number[3:5])
        day = int(id_number[5:7])
        governorate_code = id_number[7:9]
        gender_code = int(id_number[12:13])
        
        # Determine century and full year
        if century_digit == 2:
            full_year = 1900 + year
        elif century_digit == 3:
            full_year = 2000 + year
        else:
            raise ValueError(f"Invalid century digit: {century_digit}")
        
        # Determine gender
        gender = "Male" if gender_code % 2 != 0 else "Female"
        
        # Get governorate
        governorate = governorates.get(governorate_code, "Unknown")
        
        # Format birth date
        birth_date = f"{full_year:04d}-{month:02d}-{day:02d}"
        
        return {
            'Birth Date': birth_date,
            'Governorate': governorate,
            'Gender': gender
        }
    
    except (ValueError, IndexError) as e:
        print(f"Error decoding ID {id_number}: {e}")
        return {
            'Birth Date': '',
            'Governorate': '',
            'Gender': ''
        }