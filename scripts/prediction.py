def predict_with_models(X, trained_models):
    predictions = {}

    for model_name, model in trained_models.items():
        y_pred = model.predict(X)
        predictions[f"{model_name}_prediction"] = y_pred

    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.replace({1:"bad",0:"good"})
    return predictions_df