$(document).ready(function() {

/*
    var retrieved = $('#datatable-retrieved').DataTable( {
        dom: 'Bfrtip',
        //ajax: {
        //    'url': 'get_k_random',
        //},
        scrollY:        300,
        //scrollX:        true,
        scrollCollapse: true,
        paging:         false,
        fixedColumns:   {
            left: 2
        },
        columnDefs: [ {
            orderable: false,
            className: 'select-checkbox noVis',
            targets:   0
            }
        ],
        order: [[ 3, "desc" ]],
        search: {
            return: true
        },
        select: {
            // style:    'os',
            // selector: 'td:first-child'
            style: 'multi'
        },
        buttons: [
            {
                extend: 'colvis',
                columns: ':not(.noVis)',
                postfixButtons: ['colvisRestore']
            }
        ]
    } );*/


    $('#upload-new-data').on('click', function () {

       //retrieved.clear().draw();
       getData('/get_k_random').then((data) => {
           // Show
           console.log(data)

           // Parse (not needed)
           //var aux = JSON.parse(data);

           // Create columns information
           var chckbx = {title: ''}
           var caux = [];
           for (let i=0; i<data.columns.length; i++) {
               caux.push({title: data.columns[i]})
           }

           // Create datatable from scratch
          var retrieved = $('#datatable-retrieved').DataTable( {
             data:           data.data,
             columns:        caux,
             autoWidth:      true,
             scrollY:        500,
             scrollX:        true,
             scrollCollapse: true,
             paging:         false,
             fixedColumns:   {
                left: 1
             },
             //order: [[ 3, "desc" ]],
             search: {
                return: true
             },
             select: {
                //style:    'os',
                //selector: 'td:first-child',
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
                     targets: 0,
                     orderable: false,
                     className: 'select-checkbox noVis',
                 }]
          });


           $('#datatable-retrieved tbody').on( 'click', 'tr', function () {
                $(this).toggleClass('selected');

                // Study number
                var f = $(this).children('td').eq(1).text()

                // Highlight the row.
                if ( $(this).hasClass('selected') ) {

                    // Retrieve daily information for this patient and
                    // plot the trace/trajectory over the latent space.
                    // Get feature name
                    // Define url
                    const domain = document.location.origin;
                    const trace_url = domain + "/get_trace";
                    let query_url = new URL(trace_url);
                    query_url.searchParams.append('study_no', f);

                    // Query data and plot
                    getData(query_url).then((data) => {

                        plotlyTrajectory(
                            'latentSpace',
                            data.x,
                            data.y,
                            data.study_no)
                    });

                } else {
                    deleteTraces('latentSpace', f)
                }
           } );


           //$('#button').click( function () {
       // alert( table.rows('.selected').data().length +' row(s) selected' );
    //} );


           //retrieved.columns.adjust().draw()

       });
    });
});

