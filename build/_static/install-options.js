
$(document).ready(function () {

    function label(lbl) {
        return $.trim(lbl.replace(/[ .]/g, '-').toLowerCase());
    }

    // find the user os, and set the according option to active
    function setActiveOSButton() {
        var os = "linux"
        var agent = window.navigator.userAgent.toLowerCase();
        if (agent.indexOf("win") != -1) {
            os = "windows"
        } else if (agent.indexOf("mac") != -1) {
            os = "macos"
        }
        $('.install .option').each(function(){
            if (label($(this).text()).indexOf(os) != -1) {
                $(this).addClass('active');
            }
        });
    }
    setActiveOSButton();

    // apply theme
    function setTheme() {
        $('.opt-group .option').each(function(){
            $(this).addClass('mdl-button mdl-js-button mdl-js-ripple-effect mdl-button--raised ');
        });
        $('.opt-group .active').each(function(){
            $(this).addClass('mdl-button--colored');
        });
    }
    setTheme();


    // show the command according to the active options
    function showCommand() {
        $('.opt-group .option').each(function(){
            console.log(label($(this).text()));
            $('.'+label($(this).text())).hide();
        });
        $('.install .active').each(function(){
            $('.'+label($(this).text())).show();
        });
    }
    showCommand();

    function setOptions() {
        var el = $(this);
        el.siblings().removeClass('active');
        el.siblings().removeClass('mdl-button--colored');
        el.addClass('active');
        el.addClass('mdl-button--colored');
        showCommand();
    }

    $('.opt-group').on('click', '.option', setOptions);

});
