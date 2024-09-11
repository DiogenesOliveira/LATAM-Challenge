# LATAM Challenge Documentation

## About the Challenge
The challenge proposed for the Machine Learning Engineer position was to predict whether a flight is delayed or not based on a provided dataset. The challenge had 4 main stages, described in the sessions below.

## Part 1 - Building the Model
In this step it was necessary to analyze the logical reasoning of the Jupyter notebook provided and create a model.py script from it.

I decided to follow the same decisions, including using XGBoost as it is known as a robust library, capable of dealing with large volumes of data.

At this stage I had some problems such as the numpy version of require.txt (which is why I changed the version to 1.8 in the file) as well as some doubts such as whether or not it was necessary to check the time or whether the flight date was considered high season. In the end I decided to remove the methods and leave only what was necessary, leaving the code simpler and cleaner and I describe this decision better in the following section.

I also decided to use the Python pickle library to avoid direct changes to the data.csv and create a separate model depending on the project need.

## Part 2 - Building the API to the Model
The API was developed in FastAPI as requested. My initial idea was to include a few more endpoints, such as checking the high season described in the item above this document. As this was the second feature I developed, I tried to follow a TDD approach and check the API test file first and then write the code. From then on I decided that the previously created methods were not necessary and the API only had the necessary endpoints: /health and /predict.

As with Part 1, I also had problems with a library listed in requirements.txt. In this case, fastapi. Reason why I changed the version to 0.114.0.

## Part 3 - Deploying API in GCP
In this step, I tried to use GCP as suggested in README.md. Despite having never used GCP to deploy applications, I really liked the robustness and complexity of the tool.

I managed to deploy the API at the URL [URL](https://latam-api-589537469755.us-central1.run.app/docs#/) but for some reason the API stress tests did not pass due to an error in importing the ' scape' from the jinja2 library.

This was the only problem that I was unable to resolve in time and therefore, I believe it was the only point that was not successfully delivered in the challenge.

## Part 4 - Building Pipelines
This was the last feature I developed and I tried to be more objective, writing simpler pipelines, following the necessary steps for installing dependencies, building and deploying. In addition to the inclusion of patterns such as branches used in push and pull-request sessions.

I checked the Actions section on my github shortly after developing this feature to ensure that the pipelines were indeed functional.

## Conclusion
This was definitely one of the challenges I most enjoyed doing. The fact of dealing with real data, building a predictive model, creating an API and deploying the application gives a good macro idea of ​​what the work of a Machine Learning Engineer is like. It was challenging to deal with some tools for the first time like GCP and revisit some libraries that I hadn't worked on in a while, like scikitlearn and XGBoost.

In the end, I believe that the challenge was very well thought out for a backend application and made me reflect on the importance of AI in aviation and how everything is integrated so that pilots can have accurate, real-time information.
