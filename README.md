# Streamlit-based Web Application
#### EXPLORE Data Science Academy Classification Predict

## 1) Overview

![Streamlit](resources/imgs/streamlit.png)

This repository forms the basis of *Task 2* for the **Classification Predict** within EDSA's Data Science course. It hosts template code which will enable students to deploy a basic [Streamlit](https://www.streamlit.io/) web application.

As part of the predict, students are expected to expand on this base template; increasing the number of available models, user data exploration capabilities, and general Streamlit functionality.    

#### 1.1) What is Streamlit?

[![What is an API](resources/imgs/what-is-streamlit.png)](https://youtu.be/R2nr1uZ8ffc?list=PLgkF0qak9G49QlteBtxUIPapT8TzfPuB8)

If you've ever had the misfortune of having to deploy a model as an API (as was required in the Regression Sprint), you'd know that to even get basic functionality can be a tricky ordeal. Extending this framework even further to act as a web server with dynamic visuals, multiple responsive pages, and robust deployment of your models... can be a nightmare. That's where Streamlit comes along to save the day! :star:

In its own words:
> Streamlit ... is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours!  All in pure Python. All for free.

> It’s a simple and powerful app model that lets you build rich UIs incredibly quickly.

Streamlit takes away much of the background work needed in order to get a platform which can deploy your models to clients and end users. Meaning you get to focus on the important stuff (related to the data), and can largely ignore the rest. This will allow you to become a lot more productive.  

##### Description of files

For this repository, we are only concerned with a single file:

| File Name              | Description                       |
| :--------------------- | :--------------------             |
| `base_app.py`          | Streamlit application definition. |

## 2) Usage Instructions

#### 2.1) Creating a copy of this repo

| :zap: WARNING :zap:                                                                                     |
| :--------------------                                                                                   |
| Do **NOT** *clone* this repository. Instead follow the instructions in this section to *fork* the repo. |

As described within the Predict instructions for the Classification Sprint, this code represents a *template* from which to extend your own work. As such, in order to modify the template, you will need to **[fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)** this repository. Failing to do this will lead to complications when trying to work on the web application remotely.

![Fork Repo](resources/imgs/fork-repo.png)  

To fork the repo, simply ensure that you are logged into your GitHub account, and then click on the 'fork' button at the top of this page as indicated within the figure above.

#### 2.2) Running the Streamlit webapp on your local machine

As a first step to becoming familiar with our Streamlit's functioning, we recommend setting up a running instance on your own local machine.

To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

 1. Ensure that you have the prerequisite Python libraries installed on your local machine:

 ```bash
 pip install -U streamlit numpy pandas scikit-learn
 ```

 2. Clone the *forked* repo to your local machine.

 ```bash
 git clone https://github.com/{your-account-name}/classification-predict-streamlit-template.git
 ```  

 3. Navigate to the base of the cloned repo, and start the Streamlit app.

 ```bash
 cd classification-predict-streamlit-template/
 streamlit run base_app.py
 ```

 If the web server was able to initialise successfully, the following message should be displayed within your bash/terminal session:

```
  You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://192.168.43.41:8501
```

You should also be automatically directed to the base page of your web app. This should look something like:

![Streamlit base page](resources/imgs/streamlit-base-splash-screen.png)

Congratulations! You've now officially deployed your first web application!

With these steps completed, you're now ready to both modify the template code, and to host this API within an AWS EC2 instance.

While we leave the modification of your web app up to you, the latter process of cloud deployment is outlined within the next section.  

#### 2.4) Running the API on a remote AWS EC2 instance


The following steps will enable you to run your web server API on a remote EC2 instance, allowing it to the accessed by any device/application which has internet access.

Within these setup steps, we will be using a remote EC2 instance, which we will refer to as the ***Host***, in addition to our local machine, which we will call the ***Client***. We use these designations for convenience, and to align our terminology with that of common web server practices. In cases where commands are provided, use Git bash (Windows) or Terminal (Mac/Linux) to enter these.

1. Ensure that you have access to a running AWS EC2 instance with an assigned public IP address. Instructions for this process are found within the *'Introduction to Amazon AWS - Part I'* train on Athena.

2. Install the prerequisite python libraries on both the Host (EC2 instance), and Client (local machine):

```bash
pip install -U flask numpy pandas scikit-learn
```

3. Clone your copy of the API repo onto both the Host and Client machines, then navigate to the base of the cloned repo:

```bash
git clone https://github.com/{your-account-name}/regression-predict-api-template.git
cd regression-predict-api-template/
```
**[On the Host]:**

4.  Run the API web-server initialisation script.

```bash
python api.py
```

If this command ran successfully, the following output should be observed on the Host:

```
You can now view your Streamlit app in your browser.

  Network URL: http://172.31.47.109:5000
  External URL: http://3.250.50.104:5000

```

**[On the Client]:**

5.  Navigate to the `utils` subdirectory within the repo.

```bash
cd utils/
```

6. Open the `request.py` file using your favourite text editor.

    Change the value of the `url` variable to reflect the ***public IP address*** of the Host. (Instructions for getting the public IP address are provided within the *‘Introduction to Amazon AWS - Part I’* train on Athena.)

```bash
url = 'http://{public-ip-address-of-remote-machine}:5000/api_v0.1'
```   

7. Once the editing is completed, close the file and run it:

```bash
python request.py
```

 If the command ran successfully, you should see output similar to the following:

```
Sending POST request to web server API at: http://54.229.152.221:5000/api_v0.1

Querying API with the following data:
 ['Order_No_21660', 'User_Id_1329', 'Bike', 3, 'Business', 31, 5, '12:16:49 PM', 31, 5, '12:22:48 PM', 31, 5, '12:23:47 PM', 31, 5, '12:38:24 PM', 4, 21.8, nan, -1.2795183, 36.8238089, -1.273056, 36.811298, 'Rider_Id_812', 4402, 1090, 14.3, 1301]

Received POST response:
**************************************************
API prediction result: 1547.3014476106036
The response took: 0.406473 seconds
**************************************************
```

If you have completed the steps in 2.3), then the prediction result should differ from the one given above.

**[On the Host]**

You should also see an update to the web server output, indicating that it was contacted by the Client (the values of this string will differ for your output):

```
102.165.194.240 - - [08/May/2020 07:31:31] "POST /api_v0.1 HTTP/1.1" 200 -
```

If you are able to see these messages on both the Host and Client, then your API has succesfully been deployed to the Web. Snap :zap:!

## 3) FAQ

This section of the repo will be periodically updated to represent common questions which may arise around its use. If you detect any problems/bugs, please [create an issue](https://help.github.com/en/github/managing-your-work-on-github/creating-an-issue) and we will do our best to resolve it as quickly as possible.

| :information_source: NOTE :information_source: |
|:--------------------|
|You will only be able to work on this section of the API setup once you've completed the *'Introduction to Amazon AWS - Part I'* train on Athena.|

We wish you all the best in your learning experience :rocket:

![Explore Data Science Academy](resources/imgs/EDSA_logo.png)
