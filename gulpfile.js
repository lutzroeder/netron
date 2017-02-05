var gulp = require("gulp");
var sourcemaps = require('gulp-sourcemaps');
var typescript = require('gulp-typescript');
var uglify = require('gulp-uglify');

gulp.task("default", function() {
    return [
        gulp.src("./src/*.ts")
            .pipe(sourcemaps.init())
            .pipe(typescript({ target: "ES5", out: "netron.js" }))
            .pipe(uglify())
            .pipe(sourcemaps.write("."))
            .pipe(gulp.dest("./dist")),
        gulp.src([ "./samples/index.html" ])
            .pipe(gulp.dest("./dist"))
    ];
});