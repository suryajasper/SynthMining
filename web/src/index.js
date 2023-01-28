import m from 'mithril';
import './css/main.scss';
import Login from './views/login';

m.route(document.body, '/', {
  '/': Login,
  '/login': Login,
})