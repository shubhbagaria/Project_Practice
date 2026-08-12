from setuptools import find_packages,setup
from typing import List

'''this code does nothing on its own, but is only useful when something invokes it, which is done when someone runs pip install -r requirements.txt in which there is -e . and   what is does is that it runs the setup.py file.
what this file does is that first it gives pip information about the project via the informationw which is given inside the setup()'''

'''in setup() you can see find_packages() which is a function which gives the folders which are marked as importable code. NOTE THAT THE MARKING IS SEEN BY IF THE FOLDER HAS A __INIT__.PY FILE.'''

'''when we write pip install pandas, actually what happens is that this index is searched in PyPi and then the file is downloaded, but if the command pip install -e . is executed then what happens is that the . actually tells pip that there is no index this time to be searched for and downloaded, but instead setup.py file is run'''

#because now the setup.py file is run, now you can import the classes or packages from the src folder anywhere in your project.

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

setup(name='mlproject',version='0.0.1',author='Shubh',author_email='shubhbagaria1@gmail.com',packages=find_packages(),
    install_requires=get_requirements('requirements.txt'))