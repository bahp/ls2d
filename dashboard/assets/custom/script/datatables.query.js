

(function( $ ) {

	'use strict';

	var EditableTable = {

		options: {
			addButton: '#addToTable',
			submitButton: '#submitTable',
			resetButton: '#resetTable',
			table: '#datatable-editable',
			dialog: {
				wrapper: '#dialog',
				cancelButton: '#dialogCancel',
				confirmButton: '#dialogConfirm',
			},
		},

		initialize: function(columns) {

			/*This a columns example
				columns: [
					{ target:0, title: 'Day', name: 'day'},
					{ target:1, title: 'Age', name: 'age' },
					{ target:2, title: 'PLT', name: 'plt'},
					{ target:3, title: 'HCT', name: 'hct'},
					{ target:4, title: 'BT', name: 'bt'},
					{ target:5, name: 'actions', 'bSortable': false }
				],
			*/

			// Set columns
			this.columns = columns

			// Add extra column for actions
			this.columns.push({ target:5, name: 'actions', 'bSortable': false })

			this
				.setVars()
				.build()
				.events();
		},

		setVars: function() {
			// Basic
			this.$table				= $( this.options.table );
			this.$addButton			= $( this.options.addButton );
			this.$submitButton		= $( this.options.submitButton );
			this.$resetButton       = $( this.options.resetButton );

			// Dialog
			this.dialog				= {};
			this.dialog.$wrapper	= $( this.options.dialog.wrapper );
			this.dialog.$cancel		= $( this.options.dialog.cancelButton );
			this.dialog.$confirm	= $( this.options.dialog.confirmButton );

			// Return
			return this;
		},

		build: function() {

			this.datatable = this.$table.DataTable({
				searching: false,
				paging: false,
				columns: this.columns,
				responsive: true,
				scrollX: true,
				scrollCollapse: true
			});

			window.dt = this.datatable;

			return this;
		},

		events: function() {
			var _self = this;

			this.$table
				.on('click', 'a.save-row', function( e ) {
					e.preventDefault();

					_self.rowSave( $(this).closest( 'tr' ) );
				})
				.on('click', 'a.cancel-row', function( e ) {
					e.preventDefault();

					_self.rowCancel( $(this).closest( 'tr' ) );
				})
				.on('click', 'a.edit-row', function( e ) {
					e.preventDefault();

					_self.rowEdit( $(this).closest( 'tr' ) );
				})
				.on( 'click', 'a.remove-row', function( e ) {
					e.preventDefault();

					var $row = $(this).closest( 'tr' );

					$.magnificPopup.open({
						items: {
							src: '#dialog',
							type: 'inline'
						},
						preloader: false,
						modal: true,
						callbacks: {
							change: function() {
								_self.dialog.$confirm.on( 'click', function( e ) {
									e.preventDefault();

									_self.rowRemove( $row );
									$.magnificPopup.close();
								});
							},
							close: function() {
								_self.dialog.$confirm.off( 'click' );
							}
						}
					});
				});

			this.$addButton.on( 'click', function(e) {
				e.preventDefault();

				_self.rowAdd();
			});

			this.$submitButton.on( 'click', function(e) {
				e.preventDefault();

				_self.tableSubmit();
			});

			this.$resetButton.on( 'click', function(e) {
				e.preventDefault();

				_self.tableReset();
			});

			this.dialog.$cancel.on( 'click', function( e ) {
				e.preventDefault();
				$.magnificPopup.close();
			});

			return this;
		},

		// ==========================================================================================
		// ROW FUNCTIONS
		// ==========================================================================================
		rowAdd: function() {
			this.$addButton.attr({ 'disabled': 'disabled' });
			this.$submitButton.attr({ 'disabled': 'disabled' });

			var actions,
				data,
				$row;

			actions = [
				'<a href="#" class="hidden on-editing save-row"><i class="fa fa-save"></i></a>',
				'<a href="#" class="hidden on-editing cancel-row"><i class="fa fa-times"></i></a>',
				'<a href="#" class="on-default edit-row"><i class="fa fa-pencil"></i></a>',
				'<a href="#" class="on-default remove-row"><i class="fa fa-trash-o"></i></a>'
			].join(' ');

			// Create empty row.
			let n = this.datatable.init().columns.length
			let empty = Array(N).fill('')
			empty[n-1] = actions

			//data = this.datatable.row.add([ '', '', '', '', '', actions ]);
			data = this.datatable.row.add(empty);
			$row = this.datatable.row( data[0] ).nodes().to$()

			$row
				.addClass( 'adding' )
				.find( 'td:last' )
				.addClass( 'actions' );

			this.rowEdit( $row );

			this.datatable.order([0,'asc']).draw(); // always show fields
		},

		rowCancel: function( $row ) {
			var _self = this,
				$actions,
				i,
				data;

			if ( $row.hasClass('adding') ) {
				this.rowRemove( $row );
			} else {

				data = this.datatable.row( $row.get(0) ).data();
				this.datatable.row( $row.get(0) ).data( data );

				$actions = $row.find('td.actions');
				if ( $actions.get(0) ) {
					this.rowSetActionsDefault( $row );
				}

				this.datatable.draw();
			}
		},

		rowEdit: function( $row ) {
			var _self = this,
				data;

			data = this.datatable.row( $row.get(0) ).data();

			$row.children( 'td' ).each(function( i ) {
				var $this = $( this );

				if ( $this.hasClass('actions') ) {
					_self.rowSetActionsEditing( $row );
				} else {
					$this.html( '<input type="text" class="form-control input-block" value="' + data[i] + '"/>' );
				}
			});
		},

		rowSave: function( $row ) {
			var _self     = this,
				$actions,
				values    = [];

			if ( $row.hasClass( 'adding' ) ) {
				this.$addButton.removeAttr( 'disabled' );
				this.$submitButton.removeAttr( 'disabled' );
				$row.removeClass( 'adding' );
			}

			values = $row.find('td').map(function() {
				var $this = $(this);

				if ( $this.hasClass('actions') ) {
					_self.rowSetActionsDefault( $row );
					return _self.datatable.cell( this ).data();
				} else {
					return $.trim( $this.find('input').val() );
				}
			});

			this.datatable.row( $row.get(0) ).data( values );

			$actions = $row.find('td.actions');
			if ( $actions.get(0) ) {
				this.rowSetActionsDefault( $row );
			}

			this.datatable.draw();
		},

		rowRemove: function( $row ) {
			if ( $row.hasClass('adding') ) {
				this.$addButton.removeAttr( 'disabled' );
				this.$submitButton.removeAttr( 'disabled' );
			}

			this.datatable.row( $row.get(0) ).remove().draw();
		},

		rowSetActionsEditing: function( $row ) {
			$row.find( '.on-editing' ).removeClass( 'hidden' );
			$row.find( '.on-default' ).addClass( 'hidden' );
		},

		rowSetActionsDefault: function( $row ) {
			$row.find( '.on-editing' ).addClass( 'hidden' );
			$row.find( '.on-default' ).removeClass( 'hidden' );
		},

		tableReset: function() {
			this.$addButton.removeAttr( 'disabled' );
			this.$submitButton.removeAttr( 'disabled' );
			this.datatable.clear().draw();
		},

		tableSubmit: function() {
			// Get data
			var data = this.datatable.rows().data().toArray()

			// Get column names
			var names = this.columns.map(({name}) => name);

			// Convert to records
			var records = []
			for (let i = 0; i < data.length; i++) {
				records.push(Object.fromEntries(
					names.map((_, j) => [names[j], data[i][j]])))
			}

			// Sendto server
			$.ajax({
				type: "POST",
				url: 'query/patient',
				data: {
					table: JSON.stringify(records)
				},
				success: function (data) {
					// Plot trajectory
					console.log(data)
					plotlyTrajectory('latentSpace',
						data.x,
						data.y,
						data.text,
						data.study_no)
				}
			});
		}
	};

	$(function() {

		/*
        Query settings information.
		 */
		$.ajax({
			url: '/settings',
			success: function(data) {
				// Set data received as settings globally.s
				var FEATURES = JSON.parse(data.features)

				// First column will be the day
				var columns = [
					{target:0, title: 'Day', name: 'day'}
				]

				// Then add the rest of the columns
				for (var i=0; i < FEATURES.length; i++)
					columns.push({
						target: i+1,
						title: FEATURES[i].title,
						name: FEATURES[i].name
					})

				// Initialise
				EditableTable.initialize(columns);
			}
		});


		// Option I
		//EditableTable.initialize();
		// Option II
		// https://editor.datatables.net/examples/inline-editing/options.html
	});

}).apply( this, [ jQuery ]);