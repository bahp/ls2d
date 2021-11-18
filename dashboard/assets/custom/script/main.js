// from http://bl.ocks.org/mbostock/4349187
// Sample from a normal distribution with mean 0, stddev 1.

// Colorscales in plotly
// https://plotly.com/python/builtin-colorscales/

// Buttons examples in plotly
// https://plotly.com/javascript/custom-buttons/

// Tricks to improve plotly performance.
// https://www.somesolvedproblems.com/2018/07/how-do-i-make-plotly-faster.html

// Constants
const domain = document.location.origin;
const url_dataset = domain + '/get_data';
const url_knearest = domain + '/get_k_nearest';
const url_knearest_info = domain + '/get_retrieved'
const url_background = domain + '/query/column/boolean';
const url_trajectory = domain + '/enc_patient';
const url_trace = domain + '/get_trace';

let N = 0;

let SETTINGS = null;

// Global variables
var plotlyScatter = null;
var datatableDemographics = null;
var datatableRetrieved = null;
var datatableAux = null;

// Colors
var MARKER_YY = {
    color: 'rgb(102,0,0)',
    size: 3
}
var MARKER_NN = {
    color: '#749DAD',
    size: 2
}
var MARKER_Y = 'rgb(102,0,0)';
var MARKER_N = '#749DAD';

// ---------------------------------
// Define colorway
// ---------------------------------
// Define colorway which will be used when
// plotting the patient trajectories.
var COLORWAY = [
    '#f3cec9',
    '#e7a4b6',
    '#cd7eaf',
    '#a262a9',
    '#6f4d96',
    '#3d3b72',
    '#182844'
]

// ---------------------------------
// Define colormaps in dropdown menu
// ---------------------------------
// Definitions
var colormaps = [
    'YlOrRd',
    'YlGnBu',
    'Hot',
    'Reds',
    'Blues',
    'Greys',
    'Earth',
    'Electric',
    //'Viridis',
    //'Cividis'
]

// Variable
var COLORMAPS = []
for (var i = 0; i < colormaps.length; i++) {
    COLORMAPS.push({
        label: colormaps[i],
        method: 'restyle',
        args: ['colorscale', colormaps[i]]
    })
}



// ----------------------------------------------
//               Visualisation
// ----------------------------------------------
function deleteTraces(div, name) {
    /** This method deletes traces from Plotly by name.
     *
     * .. note: The trace dict needs to include the key 'name'.
     * .. note: Does it remove various traces with same name?
     *
     * Parameters
     * ----------
     * div: str
     *  The id of the container.
     * name: str
     *  The name of the trace to delete.
     */
        // Get the chart and loop
    var aux = document.getElementById(div);
    for (var i = 0; i < aux.data.length; i++) {
        if (aux.data[i]["name"] == name) {
            Plotly.deleteTraces(div, i);
        }
    }
}

function plotlyTrajectory(div, x, y, text, name) {
    /** This method plots the trajectory of a patient.
     *
     * Parameters
     * ----------
     * div: str
     *  The id of the container
     * x, y: vectors
     *  The coordinates to place the marker.
     * name: str
     *  The name of the trace
     */
    console.log("plotting..")
    console.log(x, y)
    // Trajectory
    var trace = {
            x: x,
            y: y,
            text: text,
            textposition: 'right', //'inside',
            textfont: {
                size: 12,
                family: "Arial"
                //family: "Balto"
                //family: "Droid Sans Mono"
                //family: "Droid Serif"
            },
            mode: 'lines+markers+text',
            marker: {
                size: 12,
                opacity: 1.0,
                line: {
                    width: 0.75,
                    color: 'DarkSlateGrey'
                }
            },
            line: {
                shape: 'spline',
                smoothing: 1.3
            },
            overlaying: 'points',
            name: name
        };

    // Plot
    Plotly.addTraces(div, [trace]);
}

function plotlyHistogram2dcontour(div, x, y, name) {
    /** This method plots density contours from Plotly.
     *
     * Parameters
     * ----------
     * div: str
     *  The id of the container
     * x, y: vectors
     *  The coordinates of the points to estimate density.
     * name: str
     *  The name of the trace
     */
        // Define trace
    var trace = {
            x: x,
            y: y,
            bargap: 0,
            name: 'density',
            ncontours: 14,
            colorscale: 'YlOrRd',
            reversescale: true,
            showscale: true,
            hoverinfo: 'skip',
            type: 'histogram2dcontour',
            name: name
        };

    // Delete
    deleteTraces(div, name)

    // Plot
    Plotly.addTraces(div, [trace]);
}

function plotlyHistogram2davg(div, x, y, z, name) {
    /** This method plots heatmap from Plotly.
     *
     * Parameters
     * ----------
     * div: str
     *  The id of the container
     * x, y, z: vectors
     *  The coordinates of the points and the value to average.
     * name: str
     *  The name of the trace
     */
        // Define trace
    var trace = {
            x: x,
            y: y,
            z: z,
            type: 'histogram2d',
            histfunc: 'avg',
            nbinsx: 50,
            nbinsy: 50,
            colorscale: 'Reds',
            hoverongaps: false,
            hoverinfo: 'skip',
            name: name
        }

    // Delete
    deleteTraces(div, name)

    // Plot
    Plotly.addTraces(div, [trace]);
}






// ------------------------------------------------
// Helper methods
// ------------------------------------------------

String.prototype.format = function () {
  var i = 0, args = arguments;
  return this.replace(/{}/g, function () {
    return typeof args[i] != 'undefined' ? args[i++] : '';
  });
};


async function getData(url) {
    /**
     * This method queries data.
     *
     * Parameters
     * ----------
     * url: URL
     *    The URL to query the data.
     */
    let response = await fetch(url);
    return await response.json();
}

function updateGraph() {

}

function resizePlot() {
    /*

     */
    let width = $("#latentSpace").parent().width();
    let height = $("#latentSpace").parent().height();
    Plotly.relayout('scatterPlot', {
        width: width,
        height: height
    })
}




// ----------------------------------------------
// Main
// ----------------------------------------------
// When document ready.
$(document).ready(function () {


    /*
        This piece of code is not working, probably because we
        have set the size of the figure by default to certain
        width and height. Investigate.
     */
    // Resize plot when resize.
    //window.addEventListener('resize', resizePlot);

    /*
        This piece of code is to enable the slider.
     */
    // Enable slider
    $('#listenSlider').change(function () {
        $('.output b').text(this.value);
    });
});


async function main() {
    /**
     * This is the main function.
     *
     * What it does.
     * 1. Downloads all the dataset.
     * 2. Plots the points (scatter).
     * 3. Enables clicking on points.
     *
     * REF: https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html
     *
     */
    // Retrieve data
    const scatter_data = await getData(url_dataset)

    // Information.
    console.log("Data Received:", scatter_data)

    // Global variables
    N = scatter_data.x.length;
    x = scatter_data.x;
    y = scatter_data.y;

    // Set range to total observations
    $('#selectK').slider({max: N - 1});

    // ---------------------------------------
    // Define traces
    // ---------------------------------------
    // Scatter
    var trace1 = {
        type: 'scattergl',
        mode: 'markers',
        name: 'points',
        x: scatter_data.x,
        y: scatter_data.y,
        text: scatter_data.text,
        ids: scatter_data.ids,
        marker: {
            color: MARKER_N,
            size: 2,
            opacity: 0.4,
            line: {
                width: 0
            }
        },
        hovertemplate: "(%{x}, %{y}) %{id}: %{text}"
    };

    // ------------------------------------------------------
    // Plot
    // ------------------------------------------------------
    // Layout
    var layout = {
        //title: 'Auto-Encoder - Latent Dimension',
        showlegend: false,
        autosize: true,
        //width: 500,
        //height: 500,
        //xaxis: {range: [0.1, 0.9]}, // min max
        //yaxis: {range: [0.0, 0.8]},
        margin: {
            t: 80,
            l: 50,
            r: 50,
            b: 50
        },
        hovermode: 'closest',
        bargap: 0,
        xaxis: {
            //range: [0.1, 0.9], // minmax
            //domain: [0, 0.85],
            autorange: true,
            showgrid: false,
            zeroline: false
        },
        yaxis: {
            //range: [0.0, 0.8], // minmax
            //domain: [0, 0.85],
            autorange: true,
            showgrid: false,
            zeroline: false
        },
        colorway: COLORWAY,
        updatemenus: [

            // ----------------
            // Select colormap
            // ----------------
            {
                buttons: COLORMAPS,
                pad: {'r': 0},
                x: 0.2,
                y: 1.16,
                xref: 'paper',
                yref: 'paper',
                align: 'left',
                showarrow: true
            },

            // ------------------
            // Select marker size
            // ------------------
            {
                buttons: [
                    {
                        args: [{'marker.size': 2}],
                        label: '2',
                        method: 'update'
                    },
                    {
                        args: [{'marker.size': 4}],
                        label: '4',
                        method: 'update'
                    },
                    {
                        args: [{'marker.size': 6}],
                        label: '6',
                        method: 'update'
                    },
                    {
                        args: [{'marker.size': 8}],
                        label: '8',
                        method: 'update'
                    }
                ],
                pad: {'r': 0},
                x: 4.2,
                y: 1.16,
                xref: 'paper',
                yref: 'paper',
                align: 'left',
                showarrow: true
            },

            // --------------------
            // Toggle reverse scale
            // --------------------
            {
                type: 'buttons',
                buttons: [
                    {
                        label: 'Reverse',
                        method: 'restyle',
                        args: ['reversescale', true],
                        args2: ['reversescale', false]
                    }
                ],
                pad: {'r': 0},
                x: 0.385,
                y: 1.16,
                xref: 'paper',
                yref: 'paper',
                align: 'left',
            },

            // ------------------
            // Show contour lines
            // ------------------
            {
                type: 'buttons',
                buttons: [
                    {
                        label: 'Hide lines',
                        method: 'restyle',
                        args: ['contours.showlines', true],
                        args2: ['contours.showlines', false],
                    }
                ],
                pad: {'r': 0},
                x: 0.60,
                y: 1.16,
                xref: 'paper',
                yref: 'paper',
                align: 'left',
            },

            // -------------------
            // Show legend
            // -------------------
            {
                type: 'buttons',
                buttons: [
                    {
                        label: 'Show legend',
                        method: 'relayout',
                        args: ['showlegend', false],
                        args2: ['showlegend', true]
                    }
                ],
                pad: {'r': 0},
                x: 0.85,
                y: 1.16,
                xref: 'paper',
                yref: 'paper',
                align: 'left',
            }
        ]
    };

    // Remove loading message
    $('#latentSpace').html('')

    // Plot latent space graph
    // While new plot is the common way to create a graph, seems that
    // the method react is more efficient. Both are included below for
    // reference.
    //plotlyScatter = Plotly.newPlot('latentSpace', data, layout);
    plotlyScatter = Plotly.react('latentSpace', [trace1], layout);

    // Add onOnclick handle.
    document.getElementById('latentSpace')
        .on('plotly_click', function (data) {

            // Create query
            let id = data.points[0].pointIndex;
            let k = $("#selectK").slider("option", "value");
            let query_url = new URL(url_knearest);
            query_url.searchParams.append('id', id);
            query_url.searchParams.append('k', k);

            // Update interface
            getData(query_url).then((data) => {

                // Information
                console.log("K-Nearest:", data)

                // Define colors.
                let colours = Array(N).fill(MARKER_N);
                let sizes = Array(N).fill(2);
                for (let i = 0; i < data.indexes.length; i++) {
                    colours[data.indexes[i]] = MARKER_Y; //#ff7f0e
                    sizes[data.indexes[i]] = 3;
                }

                // Update style
                Plotly.restyle('latentSpace', 'marker.color', [colours])
                Plotly.restyle('latentSpace', 'marker.size', [sizes])

                // Remove all traces but first.
                // Need to assign len to a variable because plt.data.length
                // is reevaluated on every iteration and the length of it
                // is reduced every time we delete a trace.
                var plt = document.getElementById('latentSpace');
                var len = plt.data.length;
                for (var i = 1; i < len; i++) {
                    Plotly.deleteTraces('latentSpace', 1);
                }

                // Update demographics table
                updateDemographics(data);

                // Update retrieved table
                updateKNearestInfo(data);

            });
        });
}

// Initialise
main()
