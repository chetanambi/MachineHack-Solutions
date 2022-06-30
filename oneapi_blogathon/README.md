# What is oneAPI?

There are different hardware architectures such as CPU, GPU, FPGAs, AI accelerators, etc. The code written for one architecture can’t easily run on another architecture. For example, the code written for CPU won’t run on GPU without making some changes. This is one of the problems developers face when they want to migrate their code from CPU to GPU (or FPGAs or AI accelerators).

Intel came up with a unified programming model called oneAPI to solve this very same problem. With oneAPI, it doesn't matter which hardware architectures (CPU, GPU, FGPA, or accelerators) or libraries or languages, or frameworks you use, the same code runs on all hardware architectures without any changes and additionally provides performance benefits.

*“oneAPI is a cross-industry, open, standards-based unified programming model that delivers a common developer experience across accelerator architectures — for faster application performance, more productivity, and greater innovation.*”- [oneAPI](https://www.oneapi.io/).

There are about seven oneAPI toolkits Intel provides. You can select the toolkit based on your need. I work as a data scientist hence I wanted to try out Intel® oneAPI AI Analytics Toolkit. In this article, we will mainly go through oneAPI AI Analytics toolkit.

- [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit) (for most developers)

- [Intel® oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#hpc-kit) (for HPC developers)

- **[Intel® oneAPI AI Analytics Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#analytics-kit) (for data scientists)**

- [Intel® Distribution of OpenVINO toolkit ](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#openvino-kit)(for deep learning developers)

- [Intel® oneAPI Rendering Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#rendering-kit) (for visual creators, scientists, engineers)

- [Intel® oneAPI IoT Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#iot-kit) (for edge device and IoT developers)

- [Intel® System Bring-up Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#bring-up-kit) (for system engineers)

*At first, I was reluctant to learn oneAPI because I was expecting a steep learning curve. It turns out that I was wrong. After the installation, with just a couple of lines of code changes, you can start using the toolkit immediately. You will see how easy it is to install and use it as you go through the rest of the article.*

# oneAPI AI Analytics Toolkit

The Intel oneAPI AI Analytics Toolkit (herein referred to as the toolkit) consists of many tools and frameworks. The toolkit helps data scientists to speed up their data analysis and machine learning workflow significantly.

The below diagram sums up all the tools available in the toolkit. Note that all these tools are optimized for Intel family architectures.

[![Intel oneAPI AI Analytics Toolkit contents](https://i.stack.imgur.com/Vw4pk.png "Intel oneAPI AI Analytics Toolkit contents")](https://i.stack.imgur.com/Vw4pk.png)

In the following sections, we will go through all the tools and libraries available in the toolkit. We will cover *Modin, Scikit-learn,* and *XGBoost* with examples. But we briefly touch upon *TensorFlow*, *Pytorch*, *Model Zoo*, *Neural Compressor*, and *Intel Python* as I am planning to cover these in future articles\*.\*

Note that the following code is tested in Ubuntu 20.04 LTS with Intel Core i5–6200 CPU with 16GM RAM.

## 1. Intel optimized Modin

Modin is a drop-in replacement of Pandas to speed up Pandas operations. Pandas make use of a single core whereas Modin utilizes all the cores available on the system to speed up Pandas operations.

With Intel optimized Modin, you can expect further speed improvement on Pandas operations. The icing on the cake is that you just have to make just one line of change to use it.

Run the below command from the terminal. This will create a separate environment called `aikit-modin`. The installation will take a few minutes as it has to install a lot of Intel-optimized libraries.

    conda create -n aikit-modin -c intel intel-aikit-modin`   
    conda activate aikit-modin`  
    jupyter notebook

*Note that `intel-aikit-modin` package includes **Intel® Distribution of Modin**, **Intel® Extension for Scikit-learn**, and **Intel optimizations for XGboost**. So you don’t have to install Scikit-learn and XGboost separately in the following 2 sections.*

    import numpy as np
    import modin.pandas as pddf = pd.read_csv('Reviews.csv')

## 2. Intel optimized Scikit-learn

Scikit-learn is the popular and most widely used library for implementing machine learning algorithms. The Intel optimized Scikit-learn helps to speed the model building and inference on single and multi-node systems.

If you had already installed `intel-aikit-modin` then Intel optimized Scikit-learn would have been installed already. If you wanted to use only Scikit-learn, you can use any of the below commands to install the same.

    pip install scikit-learn-intelex   
    OR  
    conda install scikit-learn-intelex -c conda-forge

In order to use the Intel optimized Scikit-learn we just need to add below 2 lines. Intel optimized Scikit-learn automatically patches the Scikit-learn algorithm to use oneAPI Data Analytics library without impacting the model performance.

    from sklearnex import patch_sklearn  
    patch_sklearn()

To undo the patching, run this line of code -

    sklearnex.unpatch_sklearn()

**Example:** In the below example, we are using the Amazon food review dataset to run a simple Naive Bayes classifier using the Intel optimized Scikit-learn.

    import pandas as pd
    
    #Intel(R) Extension for Scikit-learn dynamically patches scikit-learn estimators to use oneDAL as the underlying solver
    from sklearnex import patch_sklearn
    patch_sklearn()
    
    # Import datasets, Naive Bayes classifier and performance metrics
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    df = pd.read_csv('heart_2020_cleaned.csv')
    
    df['HeartDisease'] = df['HeartDisease'].map({'Yes': 1, 'No': 0})
    
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols) 
    
    X, y = df.drop('HeartDisease', axis=1), df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    reg = RandomForestClassifier(random_state=42)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    print("RandomForest Classifier Accuracy:",accuracy_score(y_test, preds))
    
    reg = KNeighborsClassifier()
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    print("KNN Classifier Accuracy:",accuracy_score(y_test, preds))

Output:

    RandomForest Classifier Accuracy: 0.9088669026504397
    KNN Classifier Accuracy: 0.9064903876221091

## 3. Intel optimized XGBoost

The XGBoost is one of the most widely used boosting algorithms in data science. In collaboration with the XGBoost community, Intel has optimized the XGBoost algorithm to provide high-performance w.r.t. model training and faster inference on Intel architectures. To use Intel optimized XGBoost you don't need to modify anything. You can import and use XGBoost the way use regularly. Refer to the example below.

    import xgboost as xgb
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score
    
    # Dataset: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
    df = pd.read_csv('heart_2020_cleaned.csv')
    
    df['HeartDisease'] = df['HeartDisease'].map({'Yes': 1, 'No': 0})
    
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols)    
    
    X, y = df.drop('HeartDisease', axis=1), df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    xg_reg = xgb.XGBRegressor(objective ='binary:hinge', 
                              eval_metric='logloss',
                              colsample_bytree = 0.3, 
                              learning_rate = 0.1,
                              max_depth = 5, 
                              alpha = 10, 
                              n_estimators = 10)
    
    xg_reg.fit(X_train, y_train)
    preds = xg_reg.predict(X_test)
    
    print("Accuracy:",accuracy_score(y_test, preds))

Output
Accuracy:

    Accuracy: 0.9067780710202754

## 4. Intel optimized TensorFlow & Pytorch

In collaboration with Google and Meta (Facebook), Intel has optimized the two popular deep learning libraries TensorFlow and Pytorch for Intel architectures. By using Intel-optimized TensorFlow and Pytorch, you will benefit from faster training time and inference.

The icing on the cake is that to use Intel-optimized TensorFlow and Pytorch, you don’t have to modify anything. You just need to install `intel-aikit-tensorflow` or `intel-aikit-pytorch` based on your requirement and start using the framework. As simple as that !!

    conda create -n aikit-pt -c intel intel-aikit-pytorch
    OR
    conda create -n aikit-tf -c intel intel-aikit-tensorflow

## 5. Intel optimized Python

The AI Analytics Toolkit also comes with Intel-optimized Python. When you install any of the above-mentioned tools (*Modin or TensorFlow,* or *Pytorch*), Intel optimized Python is also get installed by default.

This Intel distribution of Python includes commonly used libraries such as *Numpy, SciPy, Numba, Pandas*, and D*ata Parallel Python*. All these libraries are optimized to provide high performance which is achieved with the efficient use of multi-threading, vectorization, and more importantly memory management.

For more details on Intel distribution of Python, refer **[here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)**.

## 6. Model Zoo for Intel Architecture

Intel Model Zoo contains links to pre-trained models (such as ResNet, UNet, BERT, etc.), sample scripts, best practices, and step-by-step tutorials to run popular machine learning models on Intel architecture.

For more details on Model Zoo, refer **[here](https://github.com/IntelAI/models)**.

## 7. Intel Neural Compressor

Intel Neural Compressor is an open-source Python library that helps developers to deploy low-precision inference solutions on popular deep learning frameworks — TensorFlow, Pytorch, and ONNX.

The tool automatically optimizes low-precision recipes by applying different compression techniques such as quantization, pruning, mix-precision, etc. and thereby increasing inference performance without losing accuracy.

For more details on the Neural compressor, refer to **[this](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html)**.

# Summary

In this blog post, we went through what is oneAPI and the different toolkits available. Then we covered Intel's oneAPI AI Analytics Toolkit library with importance on Modin, Scikit-learn, and XGBoost with examples. We briefly touched upon the other tools available such as TensorFlow, Pytorch, Neural Compressor, and Model Zoo.

\#oneAPI

***Originally published on [Medium](https://pub.towardsai.net/introduction-to-intels-oneapi-ai-analytics-toolkit-8dd873925b96).***

----------

# References

\[1\]. <https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html#gs.vsqlib>

\[2\]. <https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics>
