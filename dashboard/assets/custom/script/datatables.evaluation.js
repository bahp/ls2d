(function( $ ) {

	'use strict';

	var datatableInit = function(columns, data) {

	    console.log()

	    // Select datatable
		var $table = $('#datatable-evaluation')

        // initialize
		var datatable = $table.dataTable({
            columns: columns,
            data: data,
            dom: 'Bftip',
            processing: false,
            serverSide: false,
            responsive: true,
            order: true,
            paging: false,
			colReorder: true,
			//colReorder: {
            //	order: [ 4, 3, 2, 1, 0, 5 ]
        	//},
            scrollY: 600,
            scrollX: true,
            scrollCollapse: true,
            //fixedColumns: {
            //    left: 1,
            //    right: 0
            //},
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
					targets: [2],  // slug_long
					visible: false
				},
				{
					targets: [0, 1], // estimator, slug_short
					className: 'text-bold noVis'
				},
				{
					targets: [4, 5],
					className: 'tb-demo-not-selected'
				},
				{
					targets: [11],
					className: 'tb-demo-selected'
				},
				{
					targets: 0,
					render: function (data, type, row) {
						console.log(data, type, row)
						var href = './' + row[40] + '/pipeline' + Math.trunc(row[38]) + '/pipeline' + Math.trunc(row[38]) + '-split1.p';
						console.log(href)
						//'./outputs/iris/20211216-151011/nrm-iso/pipeline1/pipeline1-split1.p'
						var a = '<a href="/model?path=' + href + '"> ' + row[0] + "</a>"
						//return row[0] + " MANUAL";
						return a;
					}
				}
			]
        });

		// format function for row details
		var fnFormatDetails = function( datatable, tr ) {
			var data = datatable.fnGetData( tr );

			return [
				'<table class="table mb-none">',
					'<tr class="b-top-none">',
						'<td><label class="mb-none">Rendering engine:</label></td>',
						'<td>' + data[1]+ ' ' + data[4] + '</td>',
					'</tr>',
					'<tr>',
						'<td><label class="mb-none">Link to source:</label></td>',
						'<td>Could provide a link here</td>',
					'</tr>',
					'<tr>',
						'<td><label class="mb-none">Extra info:</label></td>',
						'<td>And any further details here (images etc)</td>',
					'</tr>',
				'</div>'
			].join('');
		};

		// insert the expand/collapse column
		var th = document.createElement( 'th' );
		var td = document.createElement( 'td' );
		td.innerHTML = '<i data-toggle class="fa fa-plus-square-o text-primary h5 m-none" style="cursor: pointer;"></i>';
		td.className = "text-center";

		//$table
		//	.find( 'thead tr' ).each(function() {
		//	    console.log(this)
		//		this.insertBefore( th.cloneNode(true), this.childNodes[0] );
		//	});

		//$table
		//	.find( 'tbody tr' ).each(function() {
		//		this.insertBefore( td.cloneNode( true ), this.childNodes[0] );
		//	});

		// add a listener
		$table.on('click', 'i[data-toggle]', function() {
			var $this = $(this),
				tr = $(this).closest( 'tr' ).get(0);

			if ( datatable.fnIsOpen(tr) ) {
				$this.removeClass( 'fa-minus-square-o' ).addClass( 'fa-plus-square-o' );
				datatable.fnClose( tr );
			} else {
				$this.removeClass( 'fa-plus-square-o' ).addClass( 'fa-minus-square-o' );
				datatable.fnOpen( tr, fnFormatDetails( datatable, tr), 'details' );
			}
		});

		//
        datatable.columns.adjust().draw();
	};

	$(function() {

        /**
         * Ajax call!
         */
        $.ajax({
            url: 'get_evaluation',
            success: function (data) {

                // Show columns
                console.log(data)

                // This creates the list of columns that
                // will be used to create the datatable.
                var columns = []
                for (var i in data.columns) {
                    columns.push({
                        'title': data.columns[i]
                    })
                }

                // Initialise
                datatableInit(columns, data.data)

            }
        });
	});

}).apply( this, [ jQuery ]);







/*
function updateEvaluation(filepath) {

    // Information
    console.log("Updating evaluation...")

    // Create url
    var url = "get_evaluation"

    var datatableEvaluation = null;


    if (datatableEvaluation != null) {
        // Clear data
        datatableEvaluation.clear().draw()
        // Reload with new url.
        datatableEvaluation.ajax.url(url).load()

    } else {

        // Ajax column
        $.ajax({
            url: 'get_evaluation',
            success: function (data) {

                // Show columns
                console.log(data)

                // This creates the list of columns that will
                // be used to create the datatable.
                var columns = []
                for (var i in data.columns) {
                    columns.push({
                        'title': data.columns[i]
                    })
                }

                datatableInit(columns, data.data)


                // Create datatable
                datatableEvaluation = $('#datatable-evaluation').DataTable({
                    columns: columns,
                    data: data.data,
                    dom: 'Bftip',
                    processing: true,
                    serverSide: false,
                    responsive: true,
                    order: true,
                    paging: false,
                    scrollY: 1500,
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

                    //"createdRow": function ( row, data, index ) {
                    //   return '<a href="#"> Explore </a>';
                    //}


                    "columnDefs": [
                        {
                            "targets": 0,
                            "render": function (data, type, row) {
                                return '<a href="/similarity-retrieval"> Explore </a>'
                                return '<a href="#" class="btn btn-sm btn-default"> More </a>'

                            },
                        },
                    ]

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

            }
        });
    }
}

$(document).ready(function() {
    updateEvaluation('aa')
} );
*/
