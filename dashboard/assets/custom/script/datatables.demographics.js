function updateDemographics(datao) {
    /**
     * This method updates the demographics datatable.
     *
     * Parameters
     * ----------
     *
     *
     */
    // Information
    console.log("Updating demographics...")

    // Create url
    var url = "/get_demographics?idxs=" + datao.indexes.toString();

    if (datatableDemographics != null) {
        // Clear data
        datatableDemographics.clear().draw()
        // Reload with new url.
        datatableDemographics.ajax.url(url).load()

    } else {

        // Ajax column
        $.ajax({
            url: '/get_demo_columns',
            success: function (data) {

                // Show columns
                console.log(data)

                // This creates the list of columns that will be used
                // to create the datatable. Note that an empty named
                // column has been added at the beginning.
                //data.columns.unshift("")
                var columns = []
                for (var i in data.columns) {
                    columns.push({
                        'title': data.columns[i]
                    })
                }

                // Create datatable
                datatableDemographics = $('#datatable-demographics').DataTable({
                    columns: columns,
                    ajax: {
                        url: url,
                        type: 'GET',
                    },
                    dom: 'Bftip',
                    processing: true,
                    serverSide: false,
                    responsive: true,
                    order: false,
                    paging: false,
                    scrollY: 707,
                    scrollX: true,
                    scrollCollapse: true,
                    fixedColumns: {
                        left: 1,
                        right: 0
                    },
                    search: {
                        return: true
                    },
                    select: {
                        style: 'multi'
                    },
                    buttons: [
                        {
                            extend: 'colvis',
                            columns: ':not(.noVis)',
                            postfixButtons: ['colvisRestore']
                        }
                    ],
                    columnDefs: [
                        {
                            targets: [0],
                            visible: false
                        },
                        {
                            targets: [1, 2],
                            className: 'text-bold noVis'
                        },
                        {
                            targets: [5],
                            className: 'tb-demo-not-selected'
                        },
                        {
                            targets: [6],
                            className: 'tb-demo-selected'
                        },
                    ]
                });


                // Enable clicking on rows.
                $('#datatable-demographics tbody').on('click', 'tr', function () {

                    // Highlight the row.
                    if ($(this).hasClass('selected')) {
                        $(this).removeClass('selected');
                        deleteTraces('latentSpace', 'background')

                    } else {
                        datatableDemographics.$('tr.selected').removeClass('selected');
                        $(this).addClass('selected');

                        // Get feature name
                        //var f = $(this).children(':first').text().split(',')[0]

                        // Extract study number
                        //var rowIdx = datatableDemographics.row(this).index()
                        //var feature = datatableAux.cell(rowIdx, 0).data();
                        //console.log(f)

                        // Get feature name
                        var f = datatableDemographics.row(this).data()[0];

                        // Construct query
                        var query_url = new URL(url_background);
                        query_url.searchParams.append('column', f)

                        // Retrieve x, y, z
                        getData(query_url).then((data) => {
                            // Show data
                            console.log(data)

                            // Plot contour
                            if ((data.type == 'boolean')) {
                                plotlyHistogram2dcontour(
                                    'latentSpace',
                                    data.x,
                                    data.y,
                                    'background')
                            }

                            // Plot heatmap
                            if ((data.type == 'Float64') || (data.type == 'Int64')) {
                                plotlyHistogram2davg(
                                    'latentSpace',
                                    data.x,
                                    data.y,
                                    data.z,
                                    'background')
                            }
                        });
                    }
                });
            }
        });
    }
}
