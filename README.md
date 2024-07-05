# EmotionOnText: Developing a Model for Emotion Detection in Text

<!-- ![image](https://github.com/jo-walker/emotion_text_rec/assets//pictureofthemodel) -->

A lot of companies lose great employees to repressed emotional distress due to neglect of their emotional state. This can significantly affect an employee's productivity, and consequently, the overall performance of the company. Therefore, it's crucial to tune into their mental health, but directly approaching them by asking questions or pushing them to get help doesn't always work. This inefficiency can be mitigated with ML/AI technology.

This project aims to develop a model to detect emotions in text, such as employee chats from internal networks, anonymously. By analyzing these texts, we can monitor employees' emotional states and provide timely assistance, ultimately enhancing individual well-being and organizational performance.

## Table of Contents

- [Project Overview & Objective](#project-overview-&-objective)
- [Scope](#scope)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Benefits](#benefits)
- [Future Enhancements](#future-enhancements)
- [References](#references)

## Project Overview & Objective

EmotionOnText aims to revolutionize how companies manage employee well-being. The objective is to create a model that automatically analyzes the emotions expressed in employees' text communications. This model will help detect emotional distress early, allowing for timely interventions without the need for direct inquiries. By doing so, it helps prevent potential negative impacts on both individuals and the company's performance.

- check the documentation here: https://docs.google.com/document/d/18TlBY4mz9_6BliMANmsRI5WGGLWSj7tmWPVC1VD6JZU/edit?usp=sharing

## Scope

1. **Training model**: Develop and train a robust model using labeled datasets to recognize various emotions in text.
2. **Testing model**: Evaluate the model's performance using a separate test dataset to ensure accuracy.
3. **Validation model**: Validate the model on a validation dataset to fine-tune parameters and avoid overfitting.
4. **ML Algorithm**: Identify and implement the most suitable machine learning algorithms for emotion detection.
5. **Application of the model**:  Integrate the model into applications such as chatbots, web apps, or any other relevant platforms as per the company’s requirements.

## Technologies Used

- **Python**: Core programming language for model development.
- **Flask**: Web framework for developing the application.
- **spaCy**: For advanced natural language processing (NLP) tasks.
- **NLTK**: For text preprocessing and handling linguistic data.
- **scikit-learn**: For implementing ML algorithms.
- **TensorFlow & Keras**: For building DL models.

## Methodology

The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:

1. **Business Understanding**: Define the objectives and requirements from a business perspective.
2. **Data Understanding**: Collect and understand the data needed for the project.
3. **Data Preparation**: Clean and preprocess the data to make it suitable for modeling.
4. **Modeling**: Develop and train the machine learning models.
5. **Evaluation**: Assess the models to ensure they meet the business objectives.
6. **Deployment**: Implement the models in a production environment and monitor their performance.

## Benefits

1. **Efficiency**: Automatically detects emotional distress, eliminating the need for manual monitoring.
2. **Accuracy**: Provides a comprehensive analysis of emotional states based on various textual nuances.
3. **Proactivity**: Enables timely interventions to support employees before issues escalate.
4. **Anonymity**: Ensures employee privacy by analyzing text data anonymously.

## Future Enhancements

1. **Web Scraping**: Integrate web scraping capabilities to gather data from various online sources.
2. **Social Media Integration**: Extend the model to analyze emotional content from social media platforms.
3. **Recommendation System**: Develop a recommendation system to suggest resources or support based on detected emotions.
4. **Collaboration Tools**: Allow users to share emotional states and support each other, fostering a supportive community.

## References
1. Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). CARER: Contextualized Affect Representations for Emotion Recognition. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 3687-3697). Brussels, Belgium: Association for Computational Linguistics. https://www.aclweb.org/anthology/D18-1404
