var menuRight = document.getElementById( 'side-navigator' ),
				showRightPush = document.getElementById( 'showRightPush' ),
				body = document.body;

showRight.onclick = function() {
	classie.toggle( this, 'active' );
	classie.toggle( menuRight, 'cbp-spmenu-open' );
};