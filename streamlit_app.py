import streamlit as st
import requests
import json
import matplotlib.pyplot as plt

def main():
    st.title("Fake news detector v1")

    # Select model
    model_options = {
        "🤖 Fine-Tuned DistilBert Classifier": "Fine-Tuned DistilBert",
        "🧠 Convolutional Neural Network Classifier": "CNN",
        "📈 Logistic Regressor Classifier": 'Logistic Regression',
        "📚 Naive Bayes Classifier": 'Naive Bayes',
        "🗃️ Support Vector Machine Classifier": "SVM",
    }

    selected_model = st.selectbox("Select Model", list(model_options.values()))
    input_history=[]
    # User input
    st.write("Enter some text:")
    text_input = st.text_area("Input", height=150, key="text_area")
    text_input = text_input.strip()  # Trim leading and trailing whitespace

    # Predict button
    if st.button("🔍 Predict", key="predict_button"):
        if len(text_input) < 10:
            st.error("Please enter more text. Minimum 50 characters required.")
        else:
            input_data = {
                "data": text_input,
                "selected_model": selected_model
            }
            
            response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
            response_data={}
            if response.status_code == 200:
                response_data = response.json()  # Convert the response to a dictionary
                st.json(response_data)  # Display the prediction response as JSON
            
                result = response_data["result"]

                probabilities = response_data["probabilities"]
            
             # Update input history
                input_history.append((text_input, result, probabilities))
                st.session_state["input_history"] = input_history  # Store the updated input history in session state

             # Display results in a table
                table_data = {
                    "Text": [text_input],
                    "Model":[response_data['selected_model']],
                    "Prediction": [result],
                    "Probabilities": [probabilities]
                }
                st.table(table_data)

             # Show predicted class label
                st.subheader("Predicted Class")
                st.write(result)

             # Show probabilities as JSON UI
                st.subheader("Probabilities")
                st.json(probabilities)

             # Plot probabilities
                labels = list(probabilities.keys())
                values = list(probabilities.values())

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(labels, values)
                ax.set_xlabel("Class")
                ax.set_ylabel("Probability")

                st.subheader("Probability Bar Chart")
                st.pyplot(fig)
            else:
                st.error("Failed to get a response from the prediction endpoint.")

    show_history = st.button("Show History")
    if show_history:
         # Create an expander to show the full history when expanded
        with st.expander("Input History"):
             # Display the full input history in a table
            history_data = {
                "Text": [],
                "Prediction": [],
                "Probabilities": []
            }
            for item in input_history:
                history_data["Text"].append(item[0])
                history_data["Prediction"].append(item[1])
                history_data["Probabilities"].append(item[2])
            st.table(history_data)
# prit hello world
    #     # Download button
        if st.button("Download History"):
            df = pd.DataFrame(input_history, columns=["Text", "Prediction", "Probabilities"])
            file_format = st.selectbox("Select file format", ["Excel", "CSV"])

            if file_format == "Excel":
                file_extension = "xlsx"
                file_mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            else:
                file_extension = "csv"
                file_mime_type = "text/csv"

    #         # Generate a unique filename for the downloaded file
            filename = "history." + file_extension

             # Set headers for file download
            headers = {
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": file_mime_type,
            }
    #         # Stream the file for download
            st.download_button("Download", df.to_csv(index=False), file_name=filename, mime=file_mime_type, headers=headers)

if __name__ == "__main__":
    main()





# import streamlit as st
# import requests
# import json


# def main():
#     st.title("Fake news detector v1")
    
#     # Select model
#     model_options = {
#         "🤖 Fine-Tuned DistilBert Classifier": "Fine-Tuned DistilBert",
#         "🧠 Convolutional Neural Network Classifier": "CNN",
#         "📈 Logistic Regressor Classifier": 'Logistic Regression',
#         "📚 Naive Bayes Classifier": 'Naive Bayes',
#         "🗃️ Support Vector Machine Classifier": "SVM",
#     }

#     selected_model = st.selectbox("Select Model", list(model_options.values()))

#     # Load selected model
#     # model_path = model_options[selected_model]
#     # model = load_model(model_path)

#     # User input
#     st.write("Enter some text:")
#     text_input = st.text_area("Input", height=150, key="text_area")
#     text_input = text_input.strip()  # Trim leading and trailing whitespace
#     input_history = st.session_state.get("input_history", [])  # Get the existing input history

#     # Predict button
#     if st.button("🔍 Predict", key="predict_button"):
#         if len(text_input) < 10:
#             st.error("Please enter more text. Minimum 50 characters required.")
#         else:
#             # input_data = MyInputModel(data=text_input)
            
#             # q:write response simple dict
#             input_data = {
#                 "data": text_input,
#                 "selected_model": selected_model
#             }
            
#             response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

#             response_data = response.json()  # Convert the response to a dictionary
#             print("ssstream ",response)

#     #         result = response["result"]
#     #         probabilities = response["probabilities"]

#     #         # Update input history
#     #         input_history.append((text_input, result, probabilities))
#     #         st.session_state["input_history"] = input_history  # Store the updated input history in session state

#     #         # Display results in a table
#     #         table_data = {
#     #             "Text": [text_input],
#     #             "Model":[response['selected_model']],
#     #             "Prediction": [result],
#     #             "Probabilities": [probabilities]
#     #         }
#     #         st.table(table_data)

#     #         # Show predicted class label
#     #         st.subheader("Predicted Class")
#     #         st.write(response["result"])

#     #         # Show probabilities as JSON UI
#     #         st.subheader("Probabilities")
#     #         st.json(probabilities)

#     #         # Plot probabilities
#     #         labels = list(probabilities.keys())
#     #         values = list(probabilities.values())

#     #         fig, ax = plt.subplots(figsize=(6, 4))
#     #         ax.bar(labels, values)
#     #         ax.set_xlabel("Class")
#     #         ax.set_ylabel("Probability")

#     #         st.subheader("Probability Bar Chart")
#     #         st.pyplot(fig)

#     # show_history = st.button("Show History")
#     # if show_history:
#     #     # Create an expander to show the full history when expanded
#     #     with st.expander("Input History"):
#     #         # Display the full input history in a table
#     #         history_data = {
#     #             "Text": [],
#     #             "Prediction": [],
#     #             "Probabilities": []
#     #         }
#     #         for item in input_history:
#     #             history_data["Text"].append(item[0])
#     #             history_data["Prediction"].append(item[1])
#     #             history_data["Probabilities"].append(item[2])
#     #         st.table(history_data)

#     #     # Download button
#     #     if st.button("Download History"):
#     #         df = pd.DataFrame(input_history, columns=["Text", "Prediction", "Probabilities"])
#     #         file_format = st.selectbox("Select file format", ["Excel", "CSV"])

#     #         if file_format == "Excel":
#     #             file_extension = "xlsx"
#     #             file_mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#     #         else:
#     #             file_extension = "csv"
#     #             file_mime_type = "text/csv"

#     #         # Generate a unique filename for the downloaded file
#     #         filename = "history." + file_extension

#     #         # Set headers for file download
#     #         headers = {
#     #             "Content-Disposition": f"attachment; filename={filename}",
#     #             "Content-Type": file_mime_type,
#     #         }

#     #         # Stream the file for download
#     #         st.download_button("Download", df.to_csv(index=False), file_name=filename, mime=file_mime_type, headers=headers)

# if __name__ == "__main__":
#     main()