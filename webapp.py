import streamlit as st
import plotly.graph_objects as go
import numpy as np
from extended_linear_models import CubicSplines
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression


def generate_data(num_samples, seed):
    np.random.seed(seed)
    f = lambda x: np.sin(2 * x) + np.log(10 * x)
    X = np.random.uniform(0, 5, size=num_samples)
    Y_true = np.array(list(map(f, X)))
    noise = np.random.normal(0, 1, size=num_samples)
    Y_obs = Y_true + noise
    return (X, Y_obs, Y_true)


def train_spline(X, y, num_knots):
    x_max = np.max(X)
    x_min = np.min(X)
    num_regions = num_knots + 1
    k_k = lambda k: x_min + k * ((x_max - x_min) / (num_regions))
    knot_locs = [k_k(k) for k in range(1, num_regions)]
    mdl = CubicSplines(knot_locs).fit(X, y)
    return mdl


def plot_true_func(fig, X, y_true):
    zipped = zip(X, y_true)
    sorted_zipped = sorted(zipped, key=lambda x: x[0])
    sorted_x, sorted_y = zip(*sorted_zipped)
    fig.add_trace(
        go.Scatter(
            x=sorted_x,
            y=sorted_y,
            name="True Function",
            line=dict(color="#A020F0", width=3, dash="dash"),
        )
    )
    return fig


def add_plot_to_fig(fig, x, y, name, color, mode="lines"):
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name, line=dict(color=color)))
    return fig


def calc_rows_cols(num_sims):
    if num_sims < 3:
        rows = 1
        cols = num_sims
    elif num_sims <= 4:
        rows = 2
        cols = 2
    elif num_sims % 2 == 0:
        if num_sims % 3 == 0:
            rows = 2
            cols = int(num_sims / 2)
        else:
            rows = 3
            cols = 3
    else:
        rows = int(np.floor(num_sims / 2))
        cols = 3
    return (rows, cols)


def main():
    st.set_page_config("Bias-Variance Tradeoff", layout="wide")
    st.title("Bias-Variance Tradeoff Demo")
    st.markdown(
        "##### By: Dharyll Prince M. Abellana | Assistant Professor of Computer Science | University of the Philippines Cebu",
    )
    seed = 1111
    X, y_noisy, y_true = generate_data(num_samples=300, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, train_size=0.70)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train, y=y_train, mode="markers", name="Train Data"))
    fig.add_trace(go.Scatter(x=X_test, y=y_test, mode="markers", name="Test Data"))
    fig = plot_true_func(fig, X, y_true)
    colors = ["#FF8370", "#00B1B0", "#FEC84D", "#E42256"]

    with st.sidebar:
        st.markdown("Specify Number of Splines")
        num_models = st.number_input("Enter Number of Models", min_value=0, max_value=4)
        num_knots_ls = []
        if num_models > 0:
            st.markdown("## Configure Spline Parameters")
            for i in range(num_models):
                if i == 0:
                    num_knots_ls.append("None")
                elif i > 0:
                    num_knots_ls.append(
                        st.number_input(
                            f"Enter Number of Knots for Model {i+1}",
                            min_value=0,
                            max_value=20,
                        )
                    )
    tab1, tab2, tab3, tab4 = st.tabs(["Explore", "Simulate", "Variance", "Bias"])

    with tab1:
        if num_models == 0:
            st.plotly_chart(fig)
        else:
            x_min = np.min(X)
            x_max = np.max(X)
            sorted_x_train = np.sort(X_train)
            for i in range(len(num_knots_ls)):
                if i == 0:
                    mdl = LinearRegression().fit(X_train.reshape(-1, 1), y_train)
                    sorted_y_pred = mdl.predict(sorted_x_train.reshape(-1, 1)).reshape(
                        -1,
                    )
                    name = "Model 1 (linear)"
                elif i > 0:
                    mdl = train_spline(X_train, y_train, num_knots_ls[i])
                    sorted_y_pred = mdl.predict(sorted_x_train)
                    name = f"Model {i+1} (num knots={num_knots_ls[i]})"

                fig = add_plot_to_fig(
                    fig,
                    sorted_x_train,
                    sorted_y_pred,
                    color=colors[i],
                    name=name,
                )
            fig.update_layout(height=600)
            st.plotly_chart(fig)
    with st.sidebar:
        st.markdown("## Simulation")
        num_sims = st.number_input(
            "Enter Number of Simulation Iterations", min_value=1, max_value=9
        )
    with tab2:
        rows, cols = calc_rows_cols(num_sims)
        subplot_titles = tuple([f"Simulation {t+1}" for t in range(0, num_sims)])
        fig2 = make_subplots(
            rows=rows,
            cols=cols,
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
            subplot_titles=subplot_titles,
        )
        t = 0
        sim_models = []
        showlegend = True
        for i in range(rows):
            for j in range(cols):
                if t < num_sims:
                    t += 1
                    seed_sim = np.random.randint(0, 100000)
                    X_sim, y_sim, y_sim_true = generate_data(
                        num_samples=100, seed=seed_sim
                    )
                    fig2.add_trace(
                        go.Scatter(
                            x=X_sim,
                            y=y_sim,
                            mode="markers",
                            name=f"Simulation {t}",
                            showlegend=False,
                            marker={"color": "#B1D4E0"},
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
                    sim_t_mdls = []
                    for m in range(num_models):
                        if m == 0:
                            sim_mdl = LinearRegression().fit(
                                X=X_sim.reshape(-1, 1), y=y_sim
                            )
                            name = f"Model {m+1} (linear)"
                            sim_t_mdls.append(sim_mdl)
                            sorted_X_sim = np.sort(X_sim)
                            y_pred_sim = sim_mdl.predict(sorted_X_sim.reshape(-1, 1))

                        elif m > 0:
                            sim_mdl = train_spline(
                                X=X_sim, y=y_sim, num_knots=num_knots_ls[m]
                            )
                            name = f"Model {m+1} (num_knots={num_knots_ls[m]})"
                            sim_t_mdls.append(sim_mdl)
                            sorted_X_sim = np.sort(X_sim)
                            y_pred_sim = sim_mdl.predict(sorted_X_sim)
                        fig2.add_trace(
                            go.Scatter(
                                x=sorted_X_sim,
                                y=y_pred_sim,
                                mode="lines",
                                showlegend=showlegend,
                                line={"color": colors[m]},
                                name=name,
                            ),
                            row=i + 1,
                            col=j + 1,
                        )
                    sim_models.append(sim_t_mdls)
                showlegend = False

        # fig2.update_layout(xaxis=dict(automargin=True), yaxis=dict(automargin=True))
        fig2.update_layout(height=700)
        st.plotly_chart(fig2)
    with tab3:
        if num_models < 3:
            rows = 1
            cols = num_models
        else:
            rows = 2
            cols = 2

        fig3 = make_subplots(
            rows=rows,
            cols=cols,
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )
        x_sim_linspace = np.linspace(start=0.5, stop=4.5, num=100)
        m = 0
        colors_sims = []
        y_preds_avg = []
        variances_m = []
        for i in range(rows):
            for j in range(cols):
                y_preds_sim_m_ls = []
                if m < num_models:
                    y_pred_average = np.zeros(100)
                    for t in range(num_sims):
                        if m == 0:
                            y_pred_sim_m = (
                                sim_models[t][m]
                                .predict(x_sim_linspace.reshape(-1, 1))
                                .reshape(
                                    -1,
                                )
                            )
                        else:
                            y_pred_sim_m = sim_models[t][m].predict(x_sim_linspace)
                        y_preds_sim_m_ls.append(y_pred_sim_m)
                        fig3.add_trace(
                            go.Scatter(
                                x=x_sim_linspace,
                                y=y_pred_sim_m,
                                line={"color": colors[m]},
                                showlegend=False,
                                name=f"Simulation {t+1}",
                                opacity=0.6,
                            ),
                            row=i + 1,
                            col=j + 1,
                        )
                        y_pred_average = y_pred_average + y_pred_sim_m
                    y_pred_average = y_pred_average / num_sims
                    y_preds_avg.append(y_pred_average)
                    variance_m = 0
                    for n in range(len(y_pred_average)):
                        variance_n = 0
                        for t in range(num_sims):
                            variance_n += (
                                y_preds_sim_m_ls[t][n] - y_pred_average[n]
                            ) ** 2
                        variance_m += variance_n / num_sims
                    variance_m = np.round(variance_m / len(y_pred_average), 6)
                    variances_m.append(variance_m)
                    fig3.add_trace(
                        go.Scatter(
                            x=x_sim_linspace,
                            y=y_pred_average,
                            line={"color": "white", "dash": "dash", "width": 5},
                            showlegend=False,
                            name=f"Model {m+1}",
                            opacity=0.6,
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
                m += 1
        x_locs = [0.25, 0.75, 0.25, 0.75]
        y_locs = [1.05, 1.05, 0.3, 0.3]
        annotations = [
            dict(
                xref="paper",
                yref="paper",
                x=x_locs[m],
                y=y_locs[m],
                text=f"Model {m+1} (variance="
                + f"{variances_m[m] :0.3f}".rstrip("0")
                + ")",
                font=dict(size=16),
            )
            for m in range(num_models)
        ]
        fig3.update_layout(height=500, annotations=annotations)
        st.plotly_chart(fig3)

    with tab4:
        if num_models < 3:
            rows = 1
            cols = num_models
        else:
            rows = 2
            cols = 2

        x_sim_linspace = np.linspace(start=0.5, stop=4.5, num=100)
        m = 0
        f = lambda x: np.sin(2 * x) + np.log(10 * x)
        y_true_sim = np.array(list(map(f, x_sim_linspace)))
        bias_m = []
        for p in range(num_models):
            bias_i = 0
            for i in range(len(y_true_sim)):
                bias_i += (y_true_sim[i] - y_preds_avg[p][i]) ** 2
            bias_i = (1 / len(y_true_sim)) * bias_i
            bias_m.append(bias_i)
        subplot_titles = tuple(
            [
                f"Model {m+1} (bias^2=" + f"{bias_m[m] :0.3f}".rstrip("0") + ")"
                for m in range(num_models)
            ]
        )
        fig4 = make_subplots(
            rows=rows,
            cols=cols,
            horizontal_spacing=0.05,
            vertical_spacing=0.2,
            subplot_titles=subplot_titles,
        )
        showlegend = True
        for i in range(rows):
            for j in range(cols):
                if m < num_models:
                    fig4.add_trace(
                        go.Scatter(
                            x=x_sim_linspace,
                            y=y_preds_avg[m],
                            line={"color": "white", "dash": "dash", "width": 2},
                            showlegend=showlegend,
                            name="Expectation (E[theta_hat])",
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
                    fig4.add_trace(
                        go.Scatter(
                            x=x_sim_linspace,
                            y=y_true_sim,
                            line={"color": "#A020F0", "dash": "dash", "width": 2},
                            opacity=0.6,
                            showlegend=showlegend,
                            name="True (theta)",
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
                    showlegend = False
                m += 1
        fig4.update_layout(height=500)
        st.plotly_chart(fig4)
    with st.sidebar:
        st.markdown("## Mean Squared Error (MSE)")
        st.markdown(
            """
            <style>
            [data-testid="stMetricDelta"] svg {
                display: none;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # Sample data for metrics
        # metrics_data = [
        #     {"label": "Metric 1", "value": "100", "delta": "+10"},
        #     {"label": "Metric 2", "value": "200", "delta": "-5"},
        #     {"label": "Metric 3", "value": "300", "delta": "+15"},
        #     {"label": "Metric 4", "value": "400", "delta": "+20"},
        # ]
        metrics_data = [
            {
                "label": f"Model {i+1}",
                "value": np.round(bias_m[i] + variances_m[i], 3),
                "delta": "b="
                + f"{bias_m[i] :0.3f}".rstrip("0")
                + ", v="
                + f"{variances_m[i]:0.3f}".rstrip("0"),
            }
            for i in range(num_models)
        ]

        # Number of columns and rows
        num_columns = 2
        num_rows = len(metrics_data) // num_columns + (
            len(metrics_data) % num_columns > 0
        )

        # Display metrics in a matrix format
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
    try:
        main()
    except:
        pass
