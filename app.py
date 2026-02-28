import time
import streamlit as st 
import pandas as pd 
import numpy as np
import io
import seaborn as sns
import pandas.api.types as pd_types
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# page configuration for wide layout and title
st.set_page_config(page_title="AutoML Explorer", layout="wide", initial_sidebar_state="expanded")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Styled divider function
def divider(spacing="normal"):
    """Create a styled divider with optional spacing"""
    if spacing == "large":
        st.markdown("<hr style='border: 2px solid #1fd2db; margin: 30px 0;'>", unsafe_allow_html=True)
    elif spacing == "small":
        st.markdown("<hr style='border: 1px solid #1fd2db; margin: 10px 0;'>", unsafe_allow_html=True)
    else:  # normal
        st.markdown("<hr style='border: 1px solid #1fd2db; margin: 20px 0;'>", unsafe_allow_html=True)

st.markdown("<div class='section'>", unsafe_allow_html=True)
st.header("Welcome!")
st.header("We’re happy to see you on our website ❤️")

st.markdown("<h3 class='title'>You will be able to evaluate algorithm prediction accuracy in just 3 simple steps:</h3>", unsafe_allow_html=True)

st.text("1. Upload a CSV or Excel file")
st.text("2. Remove unimportant features")
st.text("3. Choose the target feature")

st.markdown("<h3 class='title'>Are you ready? Let’s get started!</h3>", unsafe_allow_html=True)
divider("large")
file_upload = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
st.markdown("</div>", unsafe_allow_html=True)




if file_upload is not None:

    # wrap data preview in box
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    if file_upload.name.endswith(".csv"):
        st.success("Your CSV file is added successfully✅")
        st.subheader("This is your data")
        df = pd.read_csv(file_upload)
        st.dataframe(df)

    elif file_upload.name.endswith((".xlsx", ".xls")):
        st.success("Your Excel file is added successfully✅")
        st.subheader("This is your data")
        df = pd.read_excel(file_upload)
        st.dataframe(df)

    else:
        st.error("Invalid file type. Please upload a CSV or Excel file.")
    st.markdown("</div>", unsafe_allow_html=True)


    
    
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    st.markdown("<h4 class='title'>🎯choose y (target) from your columns</h4>", unsafe_allow_html=True)
    select_y = st.selectbox(" ", df.columns.tolist())

    if df[select_y].isnull().sum() == 0:
        st.success(f"The target column '{select_y}' has no missing values✅")
    elif df[select_y].isnull().sum() > 0:
        st.warning(f"⚠️ The target column '{select_y}' has missing values")


    divider()


    st.markdown("<h4 class='title'>Choose columns to delete them:</h4>", unsafe_allow_html=True)

    col = df.columns.tolist()

    select_del = st.multiselect("" , col)
    df = df.drop(columns= select_del)
    st.dataframe(df)

    
    divider("large")
    st.markdown("</div>", unsafe_allow_html=True)
    col_1 = df.columns
    numeric_columns_1 = df.select_dtypes(include=["number"]).columns.tolist()
  
    str_columns_1 = df.select_dtypes(include=["object"]).columns.tolist()

    
   

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    str_columns = df.select_dtypes(include=["object"]).columns.tolist()


    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h4 class='title'>Handle missing values for categorical columns</h4>", unsafe_allow_html=True)

    nulls_categorical = df[str_columns].isnull().sum()
    
    if str_columns:  # Check if there are any missing values in categorical columns
        if nulls_categorical.sum() == 0:
            st.success("No missing values in categorical columns✅")
            
        elif nulls_categorical.sum() > 0:

            if select_y in str_columns:
                st.warning(f"⚠️ The target column '{select_y}' has missing values")

                
            st.write("Missing values in each categorical column:")
            st.write(nulls_categorical)

            cat_impute_method = st.selectbox("Choose how to handle missing values in categorical columns",
            ["Mode", "Class-based (Fill with 'Unknown')", "Drop rows with missing values"],)

            if cat_impute_method == "Mode":
                for col in str_columns:
                    mode_value = df[col].mode()[0]  # Most frequent value
                    df[col].fillna(mode_value, inplace=True)
                st.dataframe(df)    

            elif cat_impute_method == "Class-based (Fill with 'Unknown')":
                for col in str_columns:
                    df[col].fillna("Unknown", inplace=True)  # Fill with a default class
                st.dataframe(df)
            
            elif cat_impute_method == "Drop rows with missing values":
                df.dropna(subset=str_columns, inplace=True)  # Drop rows with missing values in categorical columns
                st.dataframe(df)

            st.info("Missing values in categorical columns: " + str(df[str_columns].isnull().sum().sum()))



    else:
        st.info("No categorical columns to handle🤷🏻‍♂️")

        
    
    st.markdown("</div>", unsafe_allow_html=True)
    divider()
    
    nulls_continuous = df[numeric_columns].isnull().sum()
    
    st.markdown("<h4 class='title'>Handle missing values for continuous columns</h4>", unsafe_allow_html=True)


    if numeric_columns and nulls_continuous.sum() > 0:  # Check if there are any missing values in continuous columns

        # if select_y in numeric_columns:
        #         st.warning(f"⚠️ The target column '{select_y}' has missing values")

        st.write("Missing values in each continuous column:")
        st.write(nulls_continuous)

        num_impute_method = st.selectbox(
            "Choose how to handle missing values in continuous columns",
            ["-- Select method --", "Mean", "Median", "Mode", "Drop rows with missing values"],
            index=0  # default selected is the placeholder
        )

        if num_impute_method == "-- Select method --":
            st.write(" ")

        elif num_impute_method == "Mean":
            for col in numeric_columns:
                df[col].fillna(df[col].mean(), inplace=True)
            st.dataframe(df)
        elif num_impute_method == "Median":
            for col in numeric_columns:
                df[col].fillna(df[col].median(), inplace=True)
            st.dataframe(df)
        elif num_impute_method == "Mode":
            for col in numeric_columns:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
            st.dataframe(df)
        elif num_impute_method == "Drop rows with missing values":
            df.dropna(subset=numeric_columns, inplace=True)
            st.dataframe(df)
        
        st.info("Missing values in continuous columns: " + str(df[numeric_columns].isnull().sum().sum()))

    else:
        st.success("No missing values in continuous columns✅")

    
    # divider("large")
    st.markdown("<hr style='border: 1px solid #1fd2db; margin: 20px 0;'>", unsafe_allow_html=True)

    
    
    

    # Display the DataFrame after handling missing values
    st.write("DataFrame after handling missing values:")
    st.dataframe(df)

    
    
    divider("large")
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    st.markdown("<h4 class='title'>📊 EDA on your data:</h4>", unsafe_allow_html=True)
    st.markdown("<h5 class='title'>Summary statistics</h5>", unsafe_allow_html=True)
    st.dataframe(df.describe())
    
    non_numeric = df.select_dtypes(include=['object', 'bool'])
    if not non_numeric.empty:
        st.dataframe(non_numeric.describe())
    
        
        


    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    str_columns = df.select_dtypes(include=["object"]).columns.tolist()

    divider()

    st.markdown("<h5 class='title'>Bar graph</h5>", unsafe_allow_html=True)
    if str_columns:
        selected_column = st.selectbox("Select a categorical column for bar graph", str_columns)
        st.bar_chart(df[selected_column].value_counts())

    else:
        st.warning("No categorical columns available for bar graph")

    
    divider()


    st.markdown("<h5 class='title'>Bar Graphs for All Categorical Columns</h5>", unsafe_allow_html=True)

    # Check if there are any categorical columns
    if str_columns:
        # Determine the number of columns to display side by side (for better layout)
        num_columns = len(str_columns)
        
        # Create columns dynamically based on the number of categorical columns
        # Use [1, 1, 1, ...] to evenly distribute space across all columns
        cols = st.columns(num_columns)  # This ensures we can display all bar charts side by side

        # Loop through each categorical column and display a bar graph next to each other
        for i, column in enumerate(str_columns):
            with cols[i]:  # Display the bar chart in the corresponding column
                st.subheader(f"Bar graph for {column}")
                st.bar_chart(df[column].value_counts())  # Display bar chart of value counts for each categorical column
    else:
        st.warning("No categorical columns available for bar graphs")

    
    st.markdown("<hr style='border: 1px solid #1fd2db; margin: 20px 0;'>", unsafe_allow_html=True)

    st.markdown("<h5 class='title'>Scatter graph</h5>", unsafe_allow_html=True)
    
    if numeric_columns:
        if len(numeric_columns) < 2:
            st.warning("Not enough numeric columns for scatter graph")
        
        elif len(numeric_columns) >= 2:
            selected_column_x = st.selectbox("Select X", numeric_columns)
            
            selected_column_y = st.selectbox("Select Y", numeric_columns)

            
            st.scatter_chart(df,x= selected_column_x , y=selected_column_y)
    else:
        st.write("You choose non numeric column")


    divider()

    # if len(numeric_columns) >= 2:
    #     corr = df[numeric_columns].corr()
    #     sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
    #     st.pyplot()
    # else:
    #     st.warning("Not enough numeric columns for correlation heatmap")

    



    st.markdown("<h5 class='title'>Correlation heatmap</h5>", unsafe_allow_html=True)
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr()
        
        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor("#030718")
        ax.set_facecolor("#262730")

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax,
            annot_kws={"size": 9, "color": "white"},
            xticklabels=True,
            yticklabels=True
        )

        ax.set_title("Correlation Heatmap", fontsize=15, fontweight="bold", color="white", pad=20)
        ax.tick_params(colors="white", labelsize=10)
        plt.xticks(rotation=45, ha="right", color="white")
        plt.yticks(color="white")
        
        # Make colorbar numbers white
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

        st.pyplot(fig, use_container_width=True)
    
    # st.subheader("Now you can choose columns to delete them: ")

    # col = df.columns.tolist()

    # select_del = st.multiselect("Select column to delete it, if you want :" , col)
    # df = df.drop(columns= select_del)
    # st.dataframe(df)

    
    # st.divider()
    # col_1 = df.columns
    numeric_columns_1 = df.select_dtypes(include=["number"]).columns.tolist()
  
    str_columns_1 = df.select_dtypes(include=["object"]).columns.tolist()


    divider("large")

    # st.markdown("<h4 class='title'>Choose target feature to build the model</h4>", unsafe_allow_html=True)
    
    
    
    # st.markdown("<span class='title'>Regression</span>", unsafe_allow_html=True)
    # st.markdown("<span class='title'>Classification</span>", unsafe_allow_html=True)

    # if str_columns_1:

    #     st.markdown("<h4 class='title'>Choose Encoding Type</h4>", unsafe_allow_html=True)

    #     select_encode = st.selectbox("",
    #     ["-- Select method --" , "Label Encoding", "One-Hot Encoding"])

    #     if select_encode == "Label Encoding":
    #         le = LabelEncoder()
    #         for col in str_columns_1:
    #             df[col] = le.fit_transform(df[col])
            
    #         st.dataframe(df)
        
    #     elif select_encode == "One-Hot Encoding":
    #         df = pd.get_dummies(df, columns= str_columns_1)
    #         st.dataframe(df)
            
    
    
    col_2 = df.columns
    numeric_columns_1 = df.select_dtypes(include=["number"]).columns.tolist()    
    str_columns_2 = df.select_dtypes(include=["object"]).columns.tolist()
    
    # select_y = st.selectbox("🎯choose y (target) from your columns", df.columns.tolist())
    # divider()
    st.markdown("</div>", unsafe_allow_html=True)
    # st.dataframe(df)
    
    
    


   
    # determine whether the target should be treated as classification or regression
    y = df[select_y]
    n_unique = y.nunique()
    is_classification = pd_types.is_object_dtype(y) or n_unique <= 20

    # divider("large")
    # st.markdown("<div class='section'>", unsafe_allow_html=True)


    if is_classification and st.button("🚀 Run PyCaret classification"):
        # classification task
        # st.info(f"Detected classification task ({n_unique} unique values)")
        # imbalance detection (only for classification)
        class_dist = y.value_counts()
        class_pct = (class_dist / len(y) * 100).round(1)
        if class_pct.min() < 30:
            st.warning(
                f"⚠️ Data imbalance detected: minority class {class_pct.idxmin()} only "
                f"represents {class_pct.min()}% of the dataset. SMOTE will be applied."
            )
        else:
            st.success("✅ Class distribution looks fairly balanced")

        from pycaret.classification import setup, compare_models, predict_model, pull
        experiment = setup(
            df,
            target=select_y,
            categorical_features=str_columns_2,
            session_id=42,
        )
        st.table(experiment.pull())

        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)

        best_model = compare_models(verbose=False)
        results = pull()
        st.table(results)
        predict_model(best_model, df)
        st.markdown("</div>", unsafe_allow_html=True)
        st.subheader("🔥Best classification model🔥")
        st.markdown(
            f'<span style="color: #1fd2db; font-size: 20px; font-weight: bold;">{best_model}</span>',
            unsafe_allow_html=True,
        )
        st.session_state["best_model"] = best_model

    elif not is_classification and st.button("🚀 Run PyCaret regression"):
        # regression task
        # st.info("Detected regression task")
        from pycaret.regression import setup, compare_models, predict_model, pull
        experiment = setup(df, target=select_y, session_id=42)
        st.table(experiment.pull())

        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)

        best_model = compare_models(verbose=False)
        results = pull()
        st.table(results)
        predict_model(best_model, df)
        st.subheader("🔥Best regression model🔥")
        st.markdown(
            f'<span style="color: #1fd2db; font-size: 20px; font-weight: bold;">{best_model}</span>',
            unsafe_allow_html=True,
        )
        st.session_state["best_model"] = best_model


        # …after you’ve done setup() and (optionally) shown the progress bar…
    # after pycaret search we can train a sklearn version separately
    # st.markdown("</div>", unsafe_allow_html=True)
    divider("large")
    st.markdown("<h4 class='title'>🛠 Train sklearn model and export</h4>", unsafe_allow_html=True)

    if st.button(f"🚀 Train {st.session_state.get('best_model')} by sklearn model") and st.session_state.get("best_model") is not None:
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[select_y])
        y = df[select_y]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        sklearn_model = st.session_state.get("best_model")
        # re-fit the pipeline on the training fold
        sklearn_model.fit(X_train, y_train)


        # evaluation metrics
        if is_classification:
            from sklearn.metrics import accuracy_score, classification_report
            preds = sklearn_model.predict(X_test)
            st.write("Test accuracy:", accuracy_score(y_test, preds))
            st.write("Classification report:")
            st.text(classification_report(y_test, preds))
            
        else:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            preds = sklearn_model.predict(X_test)
            st.write("R2:", r2_score(y_test, preds))
            st.write("MSE:", mean_squared_error(y_test, preds))
            st.write("MAE:", mean_absolute_error(y_test, preds))
            st.write("RMSE:", mean_squared_error(y_test, preds, squared=False))

            # scatter actual vs. predicted (dark theme) — show inside an expander
            with st.expander("Actual vs Predicted (scatter)", expanded=True):
                fig, ax = plt.subplots(figsize=(10, 4), facecolor="#030718")
                ax.set_facecolor("#262730")
                sns.scatterplot(x=y_test, y=preds, ax=ax, color="#1fd2db")
                ax.set_xlabel("Actual Values", color="white", fontsize=12)
                ax.set_ylabel("Predicted Values", color="white", fontsize=12)
                ax.set_title("Actual vs Predicted Values", color="white", fontsize=14)
                ax.tick_params(colors="white")
                legend = ax.legend(fontsize=11)
                if legend:
                    for text in legend.get_texts():
                        text.set_color("white")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)

            # line plot over index (dark theme) — show inside an expander
            with st.expander("Actual vs Predicted (line)", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 4), facecolor="#030718")
                ax.set_facecolor("#262730")
                ax.plot(y_test.values, label="Actual", color="#1fd2db")
                ax.plot(preds, label="Predicted", color="orange")
                ax.set_xlabel("Index", color="white", fontsize=12)
                ax.set_ylabel("Values", color="white", fontsize=12)
                ax.set_title("Actual vs Predicted Values", color="white", fontsize=14)
                ax.tick_params(colors="white")
                legend = ax.legend(fontsize=11)
                if legend:
                    for text in legend.get_texts():
                        text.set_color("white")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)

            # learning curve (training/validation loss)
            from sklearn.model_selection import learning_curve
            try:
                # try sklearn's learning_curve but avoid parallel workers which
                # can trigger pickling/attribute issues with complex pipelines
                n_jobs = 1
                if is_classification:
                    scoring = "accuracy"
                    invert = True
                else:
                    scoring = "neg_mean_squared_error"
                    invert = False

                train_sizes, train_scores, val_scores = learning_curve(
                    sklearn_model,
                    X,
                    y,
                    cv=5,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    train_sizes=np.linspace(0.1, 1.0, 5),
                    shuffle=True,
                    random_state=42,
                )

                if invert:
                    train_loss = 1 - np.mean(train_scores, axis=1)
                    val_loss = 1 - np.mean(val_scores, axis=1)
                else:
                    train_loss = -np.mean(train_scores, axis=1)
                    val_loss = -np.mean(val_scores, axis=1)

            except Exception as e:
                # fallback: compute a simple learning curve by training on
                # increasing fractions of the training set. This avoids
                # cloning/pickling issues that can happen with complex
                # PyCaret pipelines when sklearn tries to parallelize.
                st.warning(
                    f"Could not compute automated learning_curve(): {e}. Using fallback."
                )
                train_sizes = np.linspace(0.1, 1.0, 5)
                train_loss = []
                val_loss = []
                from sklearn.base import clone
                from sklearn.metrics import mean_squared_error, accuracy_score
                rng = np.random.RandomState(42)

                for frac in train_sizes:
                    n_samples = max(2, int(len(X_train) * frac))
                    idx = rng.choice(len(X_train), size=n_samples, replace=False)
                    X_sub = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
                    y_sub = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]

                    # try to get a fresh, unfitted copy of the estimator
                    try:
                        model_copy = clone(sklearn_model)
                    except Exception:
                        import copy

                        model_copy = copy.deepcopy(sklearn_model)

                    # fit and evaluate
                    model_copy.fit(X_sub, y_sub)
                    if invert:
                        tscore = accuracy_score(y_sub, model_copy.predict(X_sub))
                        vscore = accuracy_score(y_test, model_copy.predict(X_test))
                        train_loss.append(1 - tscore)
                        val_loss.append(1 - vscore)
                    else:
                        tscore = mean_squared_error(y_sub, model_copy.predict(X_sub))
                        vscore = mean_squared_error(y_test, model_copy.predict(X_test))
                        train_loss.append(tscore)
                        val_loss.append(vscore)

                train_loss = np.array(train_loss)
                val_loss = np.array(val_loss)

            # learning curve (dark theme) — show inside an expander
            with st.expander("Learning curve", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 4), facecolor="#030718")
                ax.set_facecolor("#262730")
                ax.plot(train_sizes, train_loss, label="train loss", marker="o", color="#1fd2db")
                ax.plot(train_sizes, val_loss, label="validation loss", marker="o", color="orange")
                ax.set_xlabel("Training set size", color="white", fontsize=12)
                ax.set_ylabel("Loss", color="white", fontsize=12)
                ax.set_title("Learning Curve", color="white", fontsize=14)
                ax.tick_params(colors="white")
                legend = ax.legend(fontsize=11)
                if legend:
                    for text in legend.get_texts():
                        text.set_color("white")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)


            

        # save and offer download
        import joblib, io
        buf = io.BytesIO()
        joblib.dump(sklearn_model, buf)
        buf.seek(0)
        st.download_button("📥 Download sklearn model", buf, file_name="sklearn_model.pkl")
        st.success("✅ Sklearn training complete and model available for download")
    # else:
    #     st.warning("Please run PyCaret first to determine the best model before training a sklearn version")



    
else: 
    st.write("Please upload CSV or Excel file.")

#end ------------------------------------------------------------------------

    # pycaret capstone 
        # 1- data reading (Done)
        # 1- apply eda (Done)
        # 2- handling missing values ... reg & cat ask user (Done)
        # 3- remove cols (Done)
        # 4- choose x,y -> detect task type (classficatio or regression) ... delete x (Done)
        # 5- catgorical data encoding. choose the user >> 1,0 or label (Done)
        # 6- pycaret 
        # in streamlit app
        

