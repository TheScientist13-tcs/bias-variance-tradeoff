import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


def f_true_func(x):
    return np.sin(2 * x) + np.log(10 * x)


def generate_data(num_samples, seed):
    np.random.seed(seed)
    X = np.random.uniform(0.5, 4.5, size=num_samples)
    Y_true = f_true_func(X)
    noise = np.random.normal(0, 1, size=num_samples)
    Y_obs = Y_true + noise
    return (X, Y_obs, Y_true)


def train_spline(X, y, num_knots):
    X_sk = X.reshape(-1, 1)
    k = max(2, num_knots)
    mdl = make_pipeline(
        SplineTransformer(n_knots=k, degree=3, knots="quantile", include_bias=True),
        LinearRegression(fit_intercept=False),
    )
    mdl.fit(X_sk, y)
    return mdl


def compute_empirical_bias_squared(y_true_vals, y_expected_preds):
    return float(np.mean((y_true_vals - y_expected_preds) ** 2))


def compute_empirical_variance(sim_preds_matrix, y_expected_preds):
    return float(np.mean((sim_preds_matrix - y_expected_preds) ** 2))


def plot_true_func(fig):
    x_smooth = np.linspace(0.5, 4.5, 500)
    y_smooth = f_true_func(x_smooth)
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            name="True Function",
            showlegend=True,
            line=dict(color="#A020F0", width=4, dash="dash"),
        )
    )
    return fig


def add_plot_to_fig(fig, x, y, name, color, mode="lines"):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=name,
            showlegend=True,
            line=dict(color=color, width=3),
        )
    )
    return fig


def main():
    st.set_page_config("Bias-Variance Tradeoff", layout="wide")

    st.title("Bias-Variance Tradeoff Demo")
    st.markdown(
        "##### By: Dharyll Prince M. Abellana | Assistant Professor of Computer Science | University of the Philippines Cebu",
    )

    with st.sidebar:
        st.markdown("## Dataset Configuration")
        n_train = st.number_input(
            "Training Sample Size (N)", min_value=10, max_value=300, value=30, step=5
        )
        max_knots = max(2, n_train - 2)
    seed = 1111
    n_test = max(15, int(n_train * 0.4))
    X, y_noisy, y_true = generate_data(num_samples=n_train + n_test, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_noisy, train_size=n_train, random_state=seed
    )

    sort_idx = np.argsort(X_train)
    X_train = X_train[sort_idx]
    y_train = y_train[sort_idx]

    colors = ["#FF8370", "#00B1B0", "#FEC84D", "#E42256"]

    with st.sidebar:
        st.markdown("## Specify Number of Splines")
        num_models = st.number_input(
            "Enter Number of Models", min_value=0, max_value=4, value=4
        )

        st.markdown("## Simulation")
        num_sims = st.number_input(
            "Enter Number of Simulation Iterations",
            min_value=1,
            max_value=100,
            value=20,
        )

        num_knots_ls = []
        if num_models > 0:
            st.markdown("## Configure Spline Parameters")
            default_knots = [2, min(5, max_knots), max_knots]
            for i in range(num_models):
                if i == 0:
                    num_knots_ls.append("None")
                elif i > 0:
                    default_k = (
                        default_knots[i - 1]
                        if (i - 1) < len(default_knots)
                        else max_knots
                    )
                    num_knots_ls.append(
                        st.number_input(
                            f"Knots for Model {i+1} (Max {max_knots} for exact fit)",
                            min_value=2,
                            max_value=max_knots,
                            value=min(default_k, max_knots),
                            key=f"knot_input_{i}",
                        )
                    )

    tab1, tab2, tab3, tab4 = st.tabs(["Explore", "Simulate", "Bias", "Variance"])

    x_plot_dense = np.linspace(np.min(X_train), np.max(X_train), 1000)
    y_min_bound = float(np.min(y_noisy) - 3)
    y_max_bound = float(np.max(y_noisy) + 3)

    # Tab 1: Explore
    with tab1:
        fig_explore = go.Figure()
        fig_explore.add_trace(
            go.Scatter(
                x=X_train,
                y=y_train,
                mode="markers",
                name="Train Data",
                showlegend=True,
                marker=dict(
                    color="#00BFFF", size=10, line=dict(width=1.5, color="black")
                ),
            )
        )
        fig_explore.add_trace(
            go.Scatter(
                x=X_test,
                y=y_test,
                mode="markers",
                name="Test Data",
                showlegend=True,
                marker=dict(
                    color="#F4D03F",
                    size=10,
                    symbol="cross",
                    line=dict(width=1.5, color="black"),
                ),
            )
        )
        fig_explore = plot_true_func(fig_explore)

        if num_models > 0:
            for i in range(len(num_knots_ls)):
                if i == 0:
                    mdl = LinearRegression().fit(X_train.reshape(-1, 1), y_train)
                    y_pred_dense = mdl.predict(x_plot_dense.reshape(-1, 1)).flatten()
                    train_mse = mean_squared_error(
                        y_train, mdl.predict(X_train.reshape(-1, 1))
                    )
                    test_mse = mean_squared_error(
                        y_test, mdl.predict(X_test.reshape(-1, 1))
                    )
                    name = f"Model 1 (Linear | Params=2) [Train MSE: {train_mse:.3f} | Test MSE: {test_mse:.3f}]"
                elif i > 0:
                    k = num_knots_ls[i]
                    mdl = train_spline(X_train, y_train, k)
                    y_pred_dense = np.clip(
                        mdl.predict(x_plot_dense.reshape(-1, 1)).flatten(), -1e5, 1e5
                    )
                    train_mse = mean_squared_error(
                        y_train, mdl.predict(X_train.reshape(-1, 1))
                    )
                    test_mse = mean_squared_error(
                        y_test, mdl.predict(X_test.reshape(-1, 1))
                    )
                    name = f"Model {i+1} (Knots={k}, Params={k+2}) [Train MSE: {train_mse:.3f} | Test MSE: {test_mse:.3f}]"

                fig_explore = add_plot_to_fig(
                    fig_explore,
                    x_plot_dense,
                    y_pred_dense,
                    color=colors[i],
                    name=name,
                )

        fig_explore.update_layout(
            height=650, yaxis_range=[y_min_bound, y_max_bound], showlegend=True
        )
        st.plotly_chart(
            fig_explore, use_container_width=True, key="explore_chart_clean"
        )

    rng = np.random.RandomState(42)
    y_train_true = f_true_func(X_train)

    exact_expected_models = []
    for m in range(num_models):
        if m == 0:
            exp_mdl = LinearRegression().fit(X_train.reshape(-1, 1), y_train_true)
        else:
            exp_mdl = train_spline(X_train, y_train_true, num_knots=num_knots_ls[m])
        exact_expected_models.append(exp_mdl)

    sim_test_mses = {m: [] for m in range(num_models)}
    sim_models = []

    for t in range(num_sims):
        y_sim_train = y_train_true + rng.normal(0, 1, size=n_train)
        y_sim_test_truth = f_true_func(X_train)
        y_sim_test = y_sim_test_truth + rng.normal(0, 1, size=n_train)

        sim_t_mdls = []
        for m in range(num_models):
            if m == 0:
                sim_mdl = LinearRegression().fit(X_train.reshape(-1, 1), y_sim_train)
            else:
                sim_mdl = train_spline(X_train, y_sim_train, num_knots=num_knots_ls[m])

            sim_t_mdls.append(sim_mdl)
            y_pred_test = sim_mdl.predict(X_train.reshape(-1, 1))
            sim_test_mses[m].append(mean_squared_error(y_sim_test, y_pred_test))

        sim_models.append(sim_t_mdls)

    avg_test_mses = [np.mean(sim_test_mses[m]) for m in range(num_models)]

    # Tab 2: Simulate
    with tab2:
        plots_to_show = min(num_sims, 12)
        cols_per_row = 4

        for batch_start in range(0, plots_to_show, cols_per_row):
            cols = st.columns(cols_per_row)
            for c_idx in range(cols_per_row):
                t_idx = batch_start + c_idx
                if t_idx < plots_to_show:
                    with cols[c_idx]:
                        st.markdown(f"**Simulation {t_idx+1}**")
                        fig_sim = go.Figure()

                        y_sim_disp = y_train_true + rng.normal(0, 1, size=n_train)
                        fig_sim.add_trace(
                            go.Scatter(
                                x=X_train,
                                y=y_sim_disp,
                                mode="markers",
                                name="Data",
                                showlegend=(t_idx == 0),
                                legendgroup="data",
                                marker={
                                    "color": "#00BFFF",
                                    "size": 4,
                                    "line": dict(width=0.8, color="black"),
                                },
                            )
                        )

                        for m in range(num_models):
                            sim_mdl = sim_models[t_idx][m]
                            model_name = (
                                f"Model {m+1} (Linear)"
                                if m == 0
                                else f"Model {m+1} (Knots={num_knots_ls[m]})"
                            )
                            y_pred_sim = np.clip(
                                sim_mdl.predict(x_plot_dense.reshape(-1, 1)).flatten(),
                                -50,
                                50,
                            )
                            fig_sim.add_trace(
                                go.Scatter(
                                    x=x_plot_dense,
                                    y=y_pred_sim,
                                    mode="lines",
                                    showlegend=(t_idx == 0),
                                    legendgroup=f"model_{m}",
                                    name=model_name,
                                    line={"color": colors[m], "width": 1.5},
                                )
                            )

                        fig_sim.update_yaxes(range=[y_min_bound, y_max_bound])
                        fig_sim.update_layout(
                            height=270,
                            margin=dict(l=5, r=5, t=5, b=5),
                            showlegend=(t_idx == 0),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                            ),
                            xaxis=dict(showticklabels=True),
                            yaxis=dict(showticklabels=True),
                        )
                        st.plotly_chart(
                            fig_sim, use_container_width=True, key=f"sim_plot_{t_idx}"
                        )

    y_expected_train_ls = []
    for m in range(num_models):
        exp_train = exact_expected_models[m].predict(X_train.reshape(-1, 1)).flatten()
        y_expected_train_ls.append(exp_train)

    bias_m = []
    for p in range(num_models):
        bias_i = compute_empirical_bias_squared(y_train_true, y_expected_train_ls[p])
        bias_m.append(np.round(bias_i, 4))

    variances_m = []
    for m in range(num_models):
        sim_preds_at_train = np.array(
            [
                sim_models[t][m].predict(X_train.reshape(-1, 1)).flatten()
                for t in range(num_sims)
            ]
        )
        var_m = compute_empirical_variance(sim_preds_at_train, y_expected_train_ls[m])
        variances_m.append(np.round(var_m, 4))

    # Tab 3: Bias
    with tab3:
        bias_cols_layout = st.columns(min(num_models, 2))
        x_true_smooth = np.linspace(np.min(X_train), np.max(X_train), 300)
        y_true_smooth = f_true_func(x_true_smooth)

        for m in range(num_models):
            with bias_cols_layout[m % 2]:
                st.markdown(f"**Model {m+1} (Bias² = {bias_m[m]:.3f})**")
                fig_bias = go.Figure()
                fig_bias.add_trace(
                    go.Scatter(
                        x=X_train,
                        y=y_expected_train_ls[m],
                        name="Expectation E[f̂(x)]",
                        showlegend=True,
                        line={"color": "#00BFFF", "dash": "dash", "width": 3.5},
                    )
                )
                fig_bias.add_trace(
                    go.Scatter(
                        x=x_true_smooth,
                        y=y_true_smooth,
                        name="True Function f(x)",
                        showlegend=True,
                        line={"color": "#A020F0", "dash": "dash", "width": 3.5},
                        opacity=0.8,
                    )
                )
                fig_bias.update_yaxes(range=[y_min_bound, y_max_bound])
                fig_bias.update_layout(
                    height=400, margin=dict(l=10, r=10, t=10, b=10), showlegend=True
                )
                st.plotly_chart(
                    fig_bias, use_container_width=True, key=f"bias_chart_{m}"
                )

    # Tab 4: Variance
    with tab4:
        var_cols_layout = st.columns(min(num_models, 2))
        for m in range(num_models):
            with var_cols_layout[m % 2]:
                st.markdown(f"**Model {m+1} (Variance = {variances_m[m]:.3f})**")
                fig_var = go.Figure()
                for t in range(num_sims):
                    fig_var.add_trace(
                        go.Scatter(
                            x=x_plot_dense,
                            y=np.clip(
                                sim_models[t][m]
                                .predict(x_plot_dense.reshape(-1, 1))
                                .flatten(),
                                -50,
                                50,
                            ),
                            line={"color": colors[m], "width": 1.2},
                            showlegend=(t == 0),
                            name="Simulated Fits",
                            opacity=0.35,
                        )
                    )
                fig_var.add_trace(
                    go.Scatter(
                        x=X_train,
                        y=y_expected_train_ls[m],
                        line={"color": "#00BFFF", "dash": "dash", "width": 4},
                        showlegend=True,
                        name="Expected Fit E[f̂(x)]",
                    )
                )
                fig_var.update_yaxes(range=[y_min_bound, y_max_bound])
                fig_var.update_layout(
                    height=400, margin=dict(l=10, r=10, t=10, b=10), showlegend=True
                )
                st.plotly_chart(fig_var, use_container_width=True, key=f"var_chart_{m}")

    with st.sidebar:
        st.markdown("## Model Evaluation Summary")
        st.markdown(
            """
            <style>
            [data-testid="stMetricValue"] {
                font-size: 1.3rem !important;
            }
            [data-testid="stMetricDelta"] {
                font-size: 0.85rem !important;
            }
            [data-testid="stMetricDelta"] svg {
                display: none;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        metrics_data = []
        for i in range(num_models):
            decomposed_epe = bias_m[i] + variances_m[i] + 1.0
            avg_empirical_test_mse = avg_test_mses[i]

            metrics_data.append(
                {
                    "label": f"Model {i+1}",
                    "value": f"D.EPE: {decomposed_epe:.3f}",
                    "delta": f"Avg Test MSE: {avg_empirical_test_mse:.3f}",
                }
            )

        num_columns = 2
        num_rows = len(metrics_data) // num_columns + (
            len(metrics_data) % num_columns > 0
        )

        for i in range(num_rows):
            cols = st.columns(num_columns)
            for j in range(num_columns):
                index = i * num_columns + j
                if index < len(metrics_data):
                    with cols[j]:
                        st.metric(
                            label=metrics_data[index]["label"],
                            value=metrics_data[index]["value"],
                            delta=metrics_data[index]["delta"],
                        )


if __name__ == "__main__":
    main()
