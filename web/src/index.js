import m from 'mithril';
import './css/main.scss';
import Login from './views/login';
import Main from './views/main';

function LoginRouter() {
  return {
    view(vnode) {
      return m(Login, {signup: false,});
    },
  };
}

function SignupRouter() {
  return {
    view(vnode) {
      return m(Login, {signup: true,});
    },
  };
}

m.route(document.body, '/', {
  '/': Login,
  '/login': LoginRouter,
  '/signup': SignupRouter,
  '/main': Main,
})