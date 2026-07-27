const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";


async function request(path) {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    {
      headers: {
        Accept: "application/json",
      },
    },
  );

  if (!response.ok) {
    let message = `API request failed with status ${response.status}`;

    try {
      const errorData = await response.json();

      if (errorData.detail) {
        message = errorData.detail;
      }
    } catch {
      // Keep the fallback message if the response is not JSON.
    }

    throw new Error(message);
  }

  return response.json();
}


export async function getDashboardData() {
  const [
    assessmentResponse,
    explanationResponse,
    historyResponse,
    modelResponse,
  ] = await Promise.all([
    request("/api/v1/us/assessment"),
    request("/api/v1/us/explanation"),
    request("/api/v1/us/stress-history"),
    request("/api/v1/us/model"),
  ]);

  return {
    assessment: assessmentResponse.assessment,
    interpretation: assessmentResponse.interpretation,
    generatedAt: assessmentResponse.generated_at_utc,

    drivers: explanationResponse.drivers,

    history: historyResponse,

    model: modelResponse.model,
    limitations: modelResponse.limitations,
  };
}