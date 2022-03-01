
function updateKNearestInfo(datao) {
    /**
     * This method updates the knearest information datatable.
     *
     * .. note: If the table does not exists it creates it by
     *          requesting first the columns and then the whole
     *          data. Otherwise, it just updates its content
     *          using ajax.
     *
     * Parameters
     * ----------
     */
    // Information
    console.log("Updating KNearest information...")

    // Create url
    //var url = 'get_retrieved?idxs={}&dsts={}'.format(
    //    datao.indexes.toString(),
    //    datao.distances.toString());

    //var url = '/get_retrieved?idxs={}'.format(
    //    datao.indexes.toString());

    var url = url_knearest_info + "?idxs=" + datao.indexes.toString();

    if (datatableAux != null) {
        // Reload with new url.
        datatableAux.ajax.url(url).load()

    } else {

        // Retrieve data
        $.ajax({url: url,
            success: function(result) {

                // Show
                console.log(result)
                var columns = result.columns;

                // Create datatable
                datatableAux = $('#datatable-knearest-info').DataTable({
                    data: result.data,
                    columns: result.columns
                        .map(elem => ({title: elem})),
                    dom: 'Bftip',
                    processing: true,
                    serverSide: false,
                    responsive: true,
                    paging: false,
                    scrollY: 500,
                    scrollX: true,
                    scrollCollapse: true,
                    order: [[1, "asc"]],
                    fixedColumns: {
                        left: 3,
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
                            collectionLayout: 'fixed two-column',
                            postfixButtons: ['colvisRestore']
                        }
                    ],
                    columnDefs: [
                        {
                            targets: 0,
                            orderable: false,
                            //className: 'select-checkbox noVis',
                        },
                        {
                            targets: [0, 1, 2],
                            className: 'text-bold'
                        },
                        {targets: [0, 1, 2, 3, 4, 5, 6, 7, 8], visible: true},
                        {targets: '_all', visible: false}
                    ]
                });


                $('#datatable-knearest-info tbody').on('click', 'tr', function () {

                    // Mark row as selected
                    $(this).toggleClass('selected');

                    // THIS DATA IS THE PARAMETER RECEIVED IN THE METHOD
                    // IT WOULD BE NICE TO DO IT IN A DIFFERENT WAY.
                    // Options: $( datatableAux.column(0).header() ).text();

                    // Extract study number
                    var rowIdx = datatableAux.row(this).index()
                    var colIdx = result.columns.findIndex(x => x === "study_no");
                    if (colIdx == -1) colIdx = 0 // make it index
                    var studyNo = datatableAux.cell(rowIdx, colIdx).data();

                    // Highlight the row.
                    if ($(this).hasClass('selected')) {

                        // Retrieve daily information for this patient and
                        // plot the trace/trajectory over the latent space.
                        // Define url
                        let query_url = new URL(url_trace);
                        query_url.searchParams.append('study_no', studyNo);

                        // Query data and plot
                        getData(query_url).then((data) => {

                            // Information
                            console.log("Trajectory:", data)

                            // Plot trajectory
                            plotlyTrajectory(
                                'latentSpace',
                                data.x,
                                data.y,
                                data.text,
                                data.study_no)
                        });

                    } else {
                        deleteTraces('latentSpace', studyNo)
                    }
                });

            }});


        /*

        // Ajax column
        $.ajax({
            url: '/get_data_columns',
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
                        //'data': data.columns[i], // if using records
                        'title': data.columns[i]
                    })
                }

                // Create datatable
                datatableAux = $('#datatable-knearest-info').DataTable({
                    columns: columns,
                    ajax: {
                        url: url,
                        type: 'GET',
                        //dataType: 'json',
                        //data: function(d) {
                        //    d.info = idxs
                        // }
                    },
                    dom: 'Bftip',
                    processing: true,
                    serverSide: false,
                    responsive: true,
                    paging: false,
                    scrollY: 500,
                    scrollX: true,
                    scrollCollapse: true,
                    order: [[ 1, "asc" ]],
                    fixedColumns: {
                        left: 3,
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
                            collectionLayout: 'fixed two-column',
                            postfixButtons: ['colvisRestore']
                        }
                    ],
                    columnDefs: [
                        {
                            targets: 0,
                            orderable: false,
                            //className: 'select-checkbox noVis',
                        },
                        {
                            targets: [0, 1, 2],
                            className: 'text-bold'
                        },
                        { targets: [0, 1, 2, 3, 4, 5, 6, 7, 8], visible: true},
                        { targets: '_all', visible: false }
                    ]
                });


                $('#datatable-knearest-info tbody').on('click', 'tr', function () {

                    // Mark row as selected
                    $(this).toggleClass('selected');

                    // THIS DATA IS THE PARAMETER RECEIVED IN THE METHOD

                    // Extract study number
                    var rowIdx = datatableAux.row(this).index()
                    var colIdx = data.columns.findIndex(x => x === "study_no");
                    var studyNo = datatableAux.cell(rowIdx, colIdx).data();

                    // Highlight the row.
                    if ($(this).hasClass('selected')) {

                        // Retrieve daily information for this patient and
                        // plot the trace/trajectory over the latent space.
                        // Define url
                        let query_url = new URL(url_trace);
                        query_url.searchParams.append('study_no', studyNo);

                        // Query data and plot
                        getData(query_url).then((data) => {

                            // Information
                            console.log("Trajectory:", data)

                            // Plot trajectory
                            plotlyTrajectory(
                                'latentSpace',
                                data.x,
                                data.y,
                                data.text,
                                data.study_no)
                        });

                    } else {
                        deleteTraces('latentSpace', studyNo)
                    }
                });
            }
        }) */
    }
}