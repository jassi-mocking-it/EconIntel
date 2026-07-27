import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BrainCircuit,
  CalendarDays,
  Gauge,
  RefreshCw,
  Server,
  ShieldCheck,
  TrendingUp,
} from "lucide-react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { getDashboardData } from "./services/api";
import "./App.css";


function formatNumber(value, digits = 1) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  return number.toFixed(digits);
}


function formatPercent(value, digits = 1) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "—";
  }

  return `${(number * 100).toFixed(digits)}%`;
}


function formatDate(dateString) {
  if (!dateString) {
    return "Unavailable";
  }

  const date = new Date(
    `${dateString}T00:00:00Z`,
  );

  return new Intl.DateTimeFormat(
    "en-US",
    {
      month: "short",
      day: "numeric",
      year: "numeric",
      timeZone: "UTC",
    },
  ).format(date);
}


function formatMonth(dateString) {
  if (!dateString) {
    return "";
  }

  const date = new Date(
    `${dateString}T00:00:00Z`,
  );

  return new Intl.DateTimeFormat(
    "en-US",
    {
      month: "short",
      year: "numeric",
      timeZone: "UTC",
    },
  ).format(date);
}


function warningTone(level) {
  const normalizedLevel = String(
    level || "",
  ).toLowerCase();

  if (
    normalizedLevel === "critical" ||
    normalizedLevel === "high"
  ) {
    return "danger";
  }

  if (
    normalizedLevel === "elevated" ||
    normalizedLevel === "guarded"
  ) {
    return "warning";
  }

  return "safe";
}


function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  tone = "neutral",
}) {
  return (
    <article className={`metric-card ${tone}`}>
      <div className="metric-icon">
        <Icon size={20} />
      </div>

      <div>
        <p className="metric-label">
          {label}
        </p>

        <p className="metric-value">
          {value}
        </p>

        <p className="metric-detail">
          {detail}
        </p>
      </div>
    </article>
  );
}


function DriverList({
  title,
  drivers,
  direction,
}) {
  const safeDrivers = Array.isArray(drivers)
    ? drivers
    : [];

  const largestContribution = Math.max(
    ...safeDrivers.map((driver) =>
      Math.abs(
        Number(driver.contribution) || 0,
      ),
    ),
    0.001,
  );

  const Icon =
    direction === "up"
      ? ArrowUpRight
      : ArrowDownRight;

  return (
    <article className="panel driver-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">
            Model explanation
          </p>

          <h3>{title}</h3>
        </div>

        <div
          className={`direction-icon ${direction}`}
        >
          <Icon size={20} />
        </div>
      </div>

      <div className="driver-list">
        {safeDrivers.length === 0 && (
          <p className="empty-message">
            No driver information is available.
          </p>
        )}

        {safeDrivers.map((driver) => {
          const contribution = Number(
            driver.contribution,
          );

          const width = Math.max(
            8,
            (
              Math.abs(contribution) /
              largestContribution
            ) * 100,
          );

          return (
            <div
              className="driver-row"
              key={`${direction}-${driver.category}`}
            >
              <div className="driver-label-row">
                <span>
                  {driver.category}
                </span>

                <strong>
                  {contribution > 0 ? "+" : ""}
                  {formatNumber(
                    contribution,
                    3,
                  )}
                </strong>
              </div>

              <div className="driver-track">
                <div
                  className={`driver-fill ${direction}`}
                  style={{
                    width: `${width}%`,
                  }}
                />
              </div>

              <span className="driver-caption">
                {driver.feature_count} related
                model features
              </span>
            </div>
          );
        })}
      </div>
    </article>
  );
}


function StressTooltip({
  active,
  payload,
  label,
}) {
  if (
    !active ||
    !payload ||
    payload.length === 0
  ) {
    return null;
  }

  const record = payload[0].payload;

  return (
    <div className="chart-tooltip">
      <strong>{formatMonth(label)}</strong>

      <div className="tooltip-row">
        <span>Economic stress</span>
        <b>
          {formatNumber(
            record.economic_stress,
            1,
          )}
        </b>
      </div>

      <div className="tooltip-row">
        <span>Unemployment</span>
        <b>
          {formatNumber(
            record.unemployment_rate,
            1,
          )}
          %
        </b>
      </div>

      <div className="tooltip-row">
        <span>Inflation</span>
        <b>
          {formatNumber(
            record.inflation_rate,
            1,
          )}
          %
        </b>
      </div>

      {record.crisis && (
        <div className="tooltip-crisis">
          {record.crisis_name ||
            "Historical crisis period"}
        </div>
      )}
    </div>
  );
}


function getVisibleSeries(series, range) {
  if (
    !Array.isArray(series) ||
    series.length === 0 ||
    range === "ALL"
  ) {
    return series || [];
  }

  const years = Number(
    range.replace("Y", ""),
  );

  const lastRecord = series[
    series.length - 1
  ];

  const finalDate = new Date(
    `${lastRecord.date}T00:00:00Z`,
  );

  const cutoffDate = new Date(finalDate);

  cutoffDate.setUTCFullYear(
    cutoffDate.getUTCFullYear() - years,
  );

  return series.filter((record) => {
    const recordDate = new Date(
      `${record.date}T00:00:00Z`,
    );

    return recordDate >= cutoffDate;
  });
}


function App() {
  const [dashboard, setDashboard] =
    useState(null);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  const [range, setRange] =
    useState("10Y");


  const loadDashboard = useCallback(
    async () => {
      setLoading(true);
      setError("");

      try {
        const response =
          await getDashboardData();

        setDashboard(response);
      } catch (requestError) {
        setError(
          requestError.message ||
            "EconIntel could not load the dashboard.",
        );
      } finally {
        setLoading(false);
      }
    },
    [],
  );


useEffect(() => {
  let cancelled = false;

  getDashboardData()
    .then((response) => {
      if (!cancelled) {
        setDashboard(response);
      }
    })
    .catch((requestError) => {
      if (!cancelled) {
        setError(
          requestError.message ||
            "EconIntel could not load the dashboard.",
        );
      }
    })
    .finally(() => {
      if (!cancelled) {
        setLoading(false);
      }
    });

  return () => {
    cancelled = true;
  };
}, []);

  const visibleSeries = useMemo(() => {
    const series =
      dashboard?.history?.series || [];

    return getVisibleSeries(
      series,
      range,
    );
  }, [dashboard, range]);


  const visibleCrises = useMemo(() => {
    const crisisPeriods =
      dashboard?.history?.crisis_periods ||
      [];

    if (visibleSeries.length === 0) {
      return [];
    }

    const firstVisibleDate =
      visibleSeries[0].date;

    return crisisPeriods.filter(
      (period) =>
        period.end_date >=
        firstVisibleDate,
    );
  }, [dashboard, visibleSeries]);


  if (loading && !dashboard) {
    return (
      <main className="centered-state">
        <div className="loading-orb">
          <RefreshCw
            className="spin"
            size={30}
          />
        </div>

        <h1>Loading EconIntel</h1>

        <p>
          Connecting to the macroeconomic
          intelligence API…
        </p>
      </main>
    );
  }


  if (error && !dashboard) {
    return (
      <main className="centered-state">
        <div className="error-orb">
          <AlertTriangle size={30} />
        </div>

        <h1>Dashboard unavailable</h1>

        <p>{error}</p>

        <p className="state-hint">
          Confirm that FastAPI is running on
          port 8000.
        </p>

        <button
          className="primary-button"
          onClick={loadDashboard}
          type="button"
        >
          Try again
        </button>
      </main>
    );
  }


  const assessment =
    dashboard.assessment || {};

  const interpretation =
    dashboard.interpretation || {};

  const drivers =
    dashboard.drivers || {};

  const model =
    dashboard.model || {};

  const validation =
    model.validation || {};

  const limitations =
    dashboard.limitations || [];

  const historySummary =
    dashboard.history?.summary || {};

  const riskPercent = Number(
    assessment.risk_probability_percent,
  ) || 0;

  const thresholdPercent = Number(
    assessment.warning_threshold_percent,
  ) || 0;

  const tone = warningTone(
    assessment.warning_level,
  );


  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <BrainCircuit size={24} />
          </div>

          <div>
            <strong>EconIntel</strong>

            <span>
              Global Macro Risk Intelligence
            </span>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="api-status">
            <span className="status-dot" />
            API connected
          </div>

          <button
            className="refresh-button"
            disabled={loading}
            onClick={loadDashboard}
            type="button"
          >
            <RefreshCw
              className={
                loading ? "spin" : ""
              }
              size={17}
            />

            Refresh
          </button>
        </div>
      </header>


      <main className="dashboard">
        <section className="hero">
          <div className="hero-copy">
            <div className="hero-label">
              <span className="country-pill">
                United States
              </span>

              <span>
                Updated{" "}
                {formatDate(
                  assessment.observation_date,
                )}
              </span>
            </div>

            <h1>
              Economic early-warning
              intelligence
            </h1>

            <p>
              {interpretation.summary ||
                "EconIntel monitors economic stress and estimates the probability of further escalation."}
            </p>

            <div
              className={`warning-banner ${tone}`}
            >
              {assessment.warning_active ? (
                <AlertTriangle size={19} />
              ) : (
                <ShieldCheck size={19} />
              )}

              <span>
                {interpretation.warning_message ||
                  "Current warning status is available."}
              </span>
            </div>
          </div>

          <div className="risk-visual">
            <div
              className={`risk-gauge ${tone}`}
              style={{
                "--risk-angle": `${Math.min(
                  Math.max(
                    riskPercent,
                    0,
                  ),
                  100,
                ) * 3.6}deg`,
              }}
            >
              <div className="risk-gauge-inner">
                <span>
                  3-month escalation risk
                </span>

                <strong>
                  {formatNumber(
                    riskPercent,
                    1,
                  )}
                  %
                </strong>

                <small>
                  Alert at{" "}
                  {formatNumber(
                    thresholdPercent,
                    0,
                  )}
                  %
                </small>
              </div>
            </div>

            <div className={`risk-level ${tone}`}>
              {assessment.warning_level ||
                "Unknown"}{" "}
              risk
            </div>
          </div>
        </section>


        <section className="metric-grid">
          <MetricCard
            icon={Gauge}
            label="Current stress"
            value={formatNumber(
              assessment.current_stress,
              1,
            )}
            detail="Economic Stress Index / 100"
          />

          <MetricCard
            icon={TrendingUp}
            label="Escalation probability"
            value={`${formatNumber(
              riskPercent,
              1,
            )}%`}
            detail={`${assessment.forecast_horizon_months || 3}-month forecast horizon`}
            tone={tone}
          />

          <MetricCard
            icon={ShieldCheck}
            label="Warning status"
            value={
              assessment.warning_active
                ? "Active"
                : "Inactive"
            }
            detail={
              assessment.warning_level ||
              "Unknown"
            }
            tone={tone}
          />

          <MetricCard
            icon={CalendarDays}
            label="Observation"
            value={formatDate(
              assessment.observation_date,
            )}
            detail="Latest available macro data"
          />
        </section>


        <section className="panel chart-panel">
          <div className="panel-heading chart-heading">
            <div>
              <p className="eyebrow">
                Historical intelligence
              </p>

              <h2>
                Economic Stress History
              </h2>

              <p className="panel-description">
                Monthly stress index with
                historical crisis periods
                highlighted.
              </p>
            </div>

            <div className="range-selector">
              {["5Y", "10Y", "20Y", "ALL"].map(
                (rangeOption) => (
                  <button
                    className={
                      range === rangeOption
                        ? "active"
                        : ""
                    }
                    key={rangeOption}
                    onClick={() =>
                      setRange(rangeOption)
                    }
                    type="button"
                  >
                    {rangeOption}
                  </button>
                ),
              )}
            </div>
          </div>

          <div className="chart-summary">
            <span>
              Latest{" "}
              <strong>
                {formatNumber(
                  historySummary.latest_stress,
                  1,
                )}
              </strong>
            </span>

            <span>
              Average{" "}
              <strong>
                {formatNumber(
                  historySummary.average_stress,
                  1,
                )}
              </strong>
            </span>

            <span>
              Historical peak{" "}
              <strong>
                {formatNumber(
                  historySummary.maximum_stress,
                  1,
                )}
              </strong>
            </span>
          </div>

          <div className="chart-frame">
            <ResponsiveContainer
              height="100%"
              width="100%"
            >
              <AreaChart
                data={visibleSeries}
                margin={{
                  top: 20,
                  right: 20,
                  left: -15,
                  bottom: 5,
                }}
              >
                <defs>
                  <linearGradient
                    id="stressGradient"
                    x1="0"
                    x2="0"
                    y1="0"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor="#2dd4bf"
                      stopOpacity={0.45}
                    />

                    <stop
                      offset="100%"
                      stopColor="#2dd4bf"
                      stopOpacity={0.02}
                    />
                  </linearGradient>
                </defs>

                <CartesianGrid
                  stroke="rgba(148, 163, 184, 0.12)"
                  strokeDasharray="4 4"
                  vertical={false}
                />

                {visibleCrises.map(
                  (period) => (
                    <ReferenceArea
                      fill="#f43f5e"
                      fillOpacity={0.08}
                      key={`${period.name}-${period.start_date}`}
                      strokeOpacity={0}
                      x1={period.start_date}
                      x2={period.end_date}
                    />
                  ),
                )}

                <XAxis
                  axisLine={false}
                  dataKey="date"
                  minTickGap={45}
                  tick={{
                    fill: "#7f91aa",
                    fontSize: 12,
                  }}
                  tickFormatter={(value) =>
                    String(value).slice(0, 4)
                  }
                  tickLine={false}
                />

                <YAxis
                  axisLine={false}
                  domain={[0, 100]}
                  tick={{
                    fill: "#7f91aa",
                    fontSize: 12,
                  }}
                  tickLine={false}
                  width={45}
                />

                <Tooltip
                  content={<StressTooltip />}
                />

                <Area
                  dataKey="economic_stress"
                  fill="url(#stressGradient)"
                  isAnimationActive={false}
                  name="Economic Stress"
                  stroke="#2dd4bf"
                  strokeWidth={2.5}
                  type="monotone"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-legend">
            <span>
              <i className="stress-key" />
              Economic Stress Index
            </span>

            <span>
              <i className="crisis-key" />
              Historical crisis period
            </span>
          </div>
        </section>


        <section className="driver-grid">
          <DriverList
            direction="up"
            drivers={
              drivers.categories_increasing_risk
            }
            title="Factors increasing risk"
          />

          <DriverList
            direction="down"
            drivers={
              drivers.categories_reducing_risk
            }
            title="Factors reducing risk"
          />
        </section>


        <section className="bottom-grid">
          <article className="panel model-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">
                  Model transparency
                </p>

                <h2>
                  Early-warning model
                </h2>
              </div>

              <div className="model-icon">
                <BrainCircuit size={22} />
              </div>
            </div>

            <div className="model-details">
              <div>
                <span>Selected model</span>
                <strong>
                  {model.name ||
                    "Logistic Regression"}
                </strong>
              </div>

              <div>
                <span>Feature count</span>
                <strong>
                  {model.feature_count || 42}
                </strong>
              </div>

              <div>
                <span>Validation</span>
                <strong>
                  Walk-forward
                </strong>
              </div>

              <div>
                <span>Decision threshold</span>
                <strong>
                  {formatPercent(
                    model.decision_threshold,
                    0,
                  )}
                </strong>
              </div>
            </div>

            <div className="validation-grid">
              <div>
                <span>Recall</span>
                <strong>
                  {formatPercent(
                    validation.average_recall,
                  )}
                </strong>
              </div>

              <div>
                <span>Precision</span>
                <strong>
                  {formatPercent(
                    validation.average_precision,
                  )}
                </strong>
              </div>

              <div>
                <span>F1 score</span>
                <strong>
                  {formatNumber(
                    validation.average_f1,
                    3,
                  )}
                </strong>
              </div>

              <div>
                <span>PR-AUC</span>
                <strong>
                  {formatNumber(
                    validation.average_pr_auc,
                    3,
                  )}
                </strong>
              </div>
            </div>

            <p className="model-target">
              <strong>Target:</strong>{" "}
              {model.target}
            </p>
          </article>


          <article className="panel limitations-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">
                  Responsible use
                </p>

                <h2>
                  Model limitations
                </h2>
              </div>

              <div className="warning-icon">
                <AlertTriangle size={22} />
              </div>
            </div>

            <ul className="limitations-list">
              {limitations.map(
                (limitation) => (
                  <li key={limitation}>
                    {limitation}
                  </li>
                ),
              )}
            </ul>
          </article>
        </section>


        <footer className="footer">
          <div>
            <Server size={16} />
            Live data supplied by the
            EconIntel FastAPI service
          </div>

          <span>
            Experimental macroeconomic
            intelligence—not financial advice.
          </span>
        </footer>
      </main>
    </div>
  );
}


export default App;