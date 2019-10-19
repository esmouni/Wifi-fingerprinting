# Wifi-fingerprinting
First contribution to the wifi indoor localisation by WiFi fingerprinting competition

1.	Dataset
The UJIIndoorLoc database covers three buildings of Universitat Jaume I with 4 or more floors and almost 110.000 m2. It was created in 2013 by means of more than 20 different users and 25 Android devices. The database consists of 19,937 training/reference records (trainingData.csv file) and 1111 validation/test records (validationData.csv file).

2. Task
The task was to train a model to predict floor, latitude and longitude of a user of a device that had logged on.

3. Approach
I used a random forest ("RF") classifier to predict building and floor and a RF regressor for longitude and latitude.

4. Results
The results were promising on the training and validation sets, but were disappointing on the testset.

5. Next steps
For the next iteration I will explore the following:
- normalizing all devices
- adversarial validation before submission




