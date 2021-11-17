$(document).ready(function() {

    $('#test-init').on('click', function () {

       //retrieved.clear().draw();
       getData('/demographics').then((data) => {
           // Show
           console.log(data)

           //document.getElementById("auxtb").innerHTML = data.table


           // Create columns information
           var chckbx = {title: ''}
           var caux = [];
           for (let i=0; i<data.columns.length; i++) {
               caux.push({title: data.columns[i][1]})
           }

            // Create datatable from scratch
            var demographics = $('#datatable-demographics').DataTable( {
             data:           data.matrix,
             columns:        caux,
             autoWidth:      true,
             scrollY:        650,
             scrollX:        false,
             scrollCollapse: true,
             paging:         false,
             fixedColumns:   {
                left: 1
             },
             //order: [[ 3, "desc" ]],
                order: false,
             search: {
                return: true
             },
             select: {
                //style:    'os',
                //selector: 'td:first-child',
                style: 'single'
             },
             //buttons: [
             //   {
             //       extend: 'colvis',
             //       columns: ':not(.noVis)',
             //       postfixButtons: ['colvisRestore']
             //   }
             //],
                columnDefs: [
                    {
                        targets: [0, 1],
                        className: 'text-bold'
                    }
                ]
             //columnDefs: [
             //    {
             //        targets: 0,
             //        orderable: false,
             //        className: 'select-checkbox noVis',
             //    }]
          });


           $('#datatable-demographics tbody').on( 'click', 'tr', function () {

               // Highlight the row.
               if ( $(this).hasClass('selected') ) {
                   $(this).removeClass('selected');
                   deleteTraces('latentSpace', 'background')

               } else {
                   demographics.$('tr.selected').removeClass('selected');
                   $(this).addClass('selected');

                   // Get feature name
                   var f = $(this).children(':first').text().split(',')[0]


                   // Construct query
                   var query = new URL(document.location.origin + '/query/column/boolean')
                   query.searchParams.append('column', f)

                   // Retrieve x, y, z
                   getData(query).then((data) => {

                       // Plot contour
                       if ((data.type=='bool')) {
                           plotlyHistogram2dcontour(
                               'latentSpace',
                               data.x,
                               data.y,
                               'background')
                       }

                       // Plot heatmap
                       if ((data.type=='float64') || (data.type=='int64')) {
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
       });
    });
});

