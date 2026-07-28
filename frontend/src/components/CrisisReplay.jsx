import {
  History,
  RotateCcw,
} from "lucide-react";


function formatReplayDate(dateString) {
  if (!dateString) {
    return "Unknown";
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


function CrisisReplay({
  crisisPeriods,
  selectedCrisis,
  onSelect,
  onReset,
}) {
  const periods = Array.isArray(
    crisisPeriods,
  )
    ? [...crisisPeriods].reverse()
    : [];

  return (
    <section className="panel replay-panel">
      <div className="panel-heading replay-heading">
        <div>
          <p className="eyebrow">
            Historical replay
          </p>

          <h2>
            Explore past stress events
          </h2>

          <p className="panel-description">
            Select a crisis to inspect economic
            conditions before, during and after
            the event.
          </p>
        </div>

        {selectedCrisis && (
          <button
            className="replay-reset-button"
            onClick={onReset}
            type="button"
          >
            <RotateCcw size={16} />
            Return to live view
          </button>
        )}
      </div>

      {periods.length === 0 ? (
        <p className="empty-message">
          No historical crisis periods are
          available.
        </p>
      ) : (
        <div className="replay-grid">
          {periods.map((period) => {
            const active =
              selectedCrisis?.name ===
                period.name &&
              selectedCrisis?.start_date ===
                period.start_date;

            return (
              <button
                className={`replay-card ${
                  active ? "active" : ""
                }`}
                key={`${period.name}-${period.start_date}`}
                onClick={() =>
                  onSelect(period)
                }
                type="button"
              >
                <div className="replay-icon">
                  <History size={18} />
                </div>

                <div>
                  <strong>
                    {period.name}
                  </strong>

                  <span>
                    {formatReplayDate(
                      period.start_date,
                    )}
                    {" — "}
                    {formatReplayDate(
                      period.end_date,
                    )}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      )}

      {selectedCrisis && (
        <div className="replay-active-message">
          Viewing the 12 months before and six
          months after{" "}
          <strong>
            {selectedCrisis.name}
          </strong>
          .
        </div>
      )}
    </section>
  );
}


export default CrisisReplay;