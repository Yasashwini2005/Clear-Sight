# Clear-Sight


### README Description

This project focuses on detecting glaucoma from fundus images using a deep learning approach. The dataset was created by combining three different datasets—**Fundus Image**, **RIM-ONE r1 DL**, and **ORIGA Dataset**—and augmented using various techniques such as **Gamma Correction**, **Horizontal Flip**, **Vertical Flip**, and a combination of both flips. The model was trained on a **VGG-16 architecture**, achieving high accuracy in glaucoma detection. 

The project also includes a **Flask-based web application** where users can input details such as name, age, gender, and a description, and upload their fundus image. Upon submission, the application predicts whether glaucoma is present or not. All user details, along with the prediction results, are stored in a **MySQL database**, and the user is provided with their test results via email. 

This project aims to provide a user-friendly interface and efficient glaucoma detection, contributing to early diagnosis and treatment.
