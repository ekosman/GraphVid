from sklearn.mixture import GaussianMixture


def get_gmm(samples, n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(samples)
    return gmm


def gmm_predict(gmm_model, samples, invert=False):
    preds = gmm_model.predict_proba(samples) * gmm_model.weights_
    preds = preds.sum(axis=1)
    if invert:
        return 1 - preds

    return preds
