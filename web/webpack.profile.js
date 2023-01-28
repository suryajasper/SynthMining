const webpack = require('webpack');
const merge = require('webpack-merge');
const Compression = require('compression-webpack-plugin');
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;
const DeadCodePlugin = require('webpack-deadcode-plugin');
const common = require('./webpack.config');

// This configuration runs the webpack bundle analyzer.
// Run with `npm run profile`

module.exports = merge(common, {
  optimization: {
    concatenateModules: false,
  },
  plugins: [
    new DeadCodePlugin({
      patterns: [
        'src/**/*.(js|css)',
      ],
    }),
    new BundleAnalyzerPlugin(),
    new Compression(),
  ],
});
