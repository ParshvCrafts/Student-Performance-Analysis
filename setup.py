from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    """
    This function will return the list of requirements
    mentioned in the requirements.txt file

    Args:
        file_path (str): _description_

    Returns:
        List[str]: _description_
    """
    
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
    
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
    name='student_performance_predictor',
    version='1.0.0',
    author= "Parshv Patel",
    author_email= "p1a2r3s4h5v6@gmail.com",
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt')
)

