"use client";

import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { BrainVisualizationResponse } from "@/lib/types";

const Plot = createPlotlyComponent(Plotly);

export default function BrainPlot({ figure }: { figure: BrainVisualizationResponse["figure"] }) {
  return (
    <Plot
      data={figure.data as Plotly.Data[]}
      layout={figure.layout as Partial<Plotly.Layout>}
      frames={figure.frames as Plotly.Frame[]}
      config={{
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
        modeBarButtonsToRemove: [
          "toImage",
          "sendDataToCloud",
          "select2d",
          "lasso2d",
          "autoScale2d",
          "hoverClosestCartesian",
          "hoverCompareCartesian",
          "toggleSpikelines",
        ],
      }}
      useResizeHandler
      className="h-full w-full"
      style={{ width: "100%", height: "100%" }}
    />
  );
}
